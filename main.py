import cv2
import numpy as np
import random
import scipy.interpolate as si
import os
from style import Style

class Painter():
    """
    """
    def __init__(self, style):
        self.style = style

    def calculate_difference(self, canvas, referenceImage):
        r_diff = canvas[:, :, 0]-referenceImage[:, :, 0]
        g_diff = canvas[:, :, 1]-referenceImage[:, :, 1]
        b_diff = canvas[:, :, 2]-referenceImage[:, :, 2]
        return np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)
    
    def makeSplineStroke(self, R, y0, x0, referenceImage):
        K = [(y0, x0)] # a new stroke with radius R
        x, y = x0, y0
        lastDx, lastDy = 0, 0
        H, W, _ = referenceImage.shape

        ref_color = referenceImage[y0, x0] # R,G,B
        self.stroke_color = ref_color
        canvas_color = self.canvas[y0, x0]
        flag = np.linalg.norm(ref_color - canvas_color) < np.linalg.norm(ref_color - self.stroke_color)

        for i in range(1, self.style.max_stroke_len+1):
            if i>self.style.min_stroke_len and flag:
                return K
        
            # get unit vector of gradient
            gx, gy = np.sum(self.grad_x[y][x]), np.sum(self.grad_y[y][x])

            #detect vanishing gradient
            if gx == 0 or gy==0:
                return K
    
            # normal vector
            dx, dy = -gy, gx

            # if necessary, reverse direction
            if lastDx*dx + lastDy*dy < 0:
                dx, dy = -dx, -dy
            
            # filter the stroke direction
            fc = self.style.curvature_filter
            dx =  fc*dx + (1-fc)*lastDx
            dy =  fc*dy + (1-fc)*lastDy
            dx = dx/(dx**2+dy**2)**0.5
            dy = dy/(dx**2+dy**2)**0.5
            x, y = int(x+R*dx), int(y+y*dy)
            x = max(min(x, W-1), 0)
            y = max(min(y, H-1), 0)
            lastDx, lastDy = dx, dy

            K.append((y, x))

        return K
    
    def bspline(self, cv, n=100, degree=3, periodic=False):
        """ Calculate n samples on a bspline
        https://stackoverflow.com/questions/24612626/b-spline-interpolation-with-python
            cv :      Array ov control vertices
            n  :      Number of samples to return
            degree:   Curve degree
            periodic: True - Curve is closed
                      False - Curve is open
        """
        # If periodic, extend the point array by count+degree+1
        cv = np.asarray(cv)
        count = len(cv)
        if periodic:
            factor, fraction = divmod(count + degree + 1, count)
            cv = np.concatenate((cv,) * factor + (cv[:fraction],))
            count = len(cv)
            degree = np.clip(degree, 1, degree)
        # If opened, prevent degree from exceeding count-1
        else:
            degree = np.clip(degree, 1, count - 1)
        # Calculate knot vector
        kv = None
        if periodic:
            kv = np.arange(0 - degree, count + degree + degree - 1, dtype='int')
        else:
            kv = np.concatenate(([0] * degree, np.arange(count - degree + 1), [count - degree] * degree))
        # Calculate query range
        u = np.linspace(periodic, (count - degree), n)
        # Calculate result
        return np.array(si.splev(u, (kv, cv.T, degree))).T

    def draw_spline(self, K, r):
        alpha = np.zeros([self.canvas.shape[0], self.canvas.shape[1]]).astype('float32') # (H, W)
        pts = self.bspline(K, n=50, degree=3, periodic=False)
        t = 1.
        for p in pts:
            x = int(p[1])
            y = int(p[0])
            cv2.circle(alpha, (x,y), r, t, -1)
        return 1 - alpha
    
    def blend(self, canvas, alpha, stroke_color):
        """
        All arrays are in the range [0-1]
        :param canvas: [3, H, W]
        :param alpha: white background alpha of shape [H,W]
        :return:
        """
        # stroke_color = self.stroke_color # [r,g,b]
        color = np.expand_dims(stroke_color, axis=(0, 1))
        alpha = np.expand_dims(alpha, axis=-1) #(H, W, 1)
        stroke_c = (1 - alpha) * color  # (H, W, 3)
        canvas = canvas * alpha + stroke_c 
        return canvas

    def paintLayer(self, canvas, referenceImage, R):
        """
        :param canvas: numpy (H*W*C)
        :param referenceImage: reference image for this layer (H*W*C)
        :param brush_size: brush radius used in this layer (int)
        :return: painted_layer (H*W*C)
        """
        S = {} #empty strokes

        # create a pointwise difference image
        D = self.calculate_difference(canvas, referenceImage)

        grid = max(int(self.style.grid_size * R), 1) #grid!=0
        imageHeight, imageWidth, _ = canvas.shape

        # convert to luminance 
        referenceImage_float = referenceImage.astype(np.float32)
        red = referenceImage_float[:, :, 0]
        green = referenceImage_float[:, :, 1]
        blue = referenceImage_float[:, :, 2]
        luminance = 0.30 * red + 0.59 * green + 0.11 * blue #luminance in float32
        # luminance = np.clip(Y, 0, 255).astype(np.uint8)
        self.grad_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)    
        self.grad_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3) 

        # for x in range(half, imageWidth-half+1, grid):
        #     for y in range(half, imageHeight-half+1, grid):
        #         # sum the error near (x,y)
        #         patch_x_start = max(0, x-half)
        #         patch_x_end = min(x+half, imageWidth)
        #         patch_y_start = max(0, y-half)
        #         patch_y_end = min(y+half, imageHeight)
        #         # patch_ref = referenceImage[patch_y_start:patch_y_end, patch_x_start:patch_x_end, :]
        #         # patch_canvas = canvas[patch_y_start:patch_y_end, patch_x_start:patch_x_end, :]

        #         D_patch = D[patch_y_start:patch_y_end, patch_x_start:patch_x_end, :]
        #         D_patch_mean = np.mean(D_patch, axis=2) # D_patch_mean (H*W)
        #         areaError = np.mean(D_patch_mean)

        #         if(areaError<self.style.threshold):
        #             # find the largest error point
        #             y_patch_1, x_patch_1 = np.unravel_index(np.argmax(D_patch_mean), D_patch_mean.shape)
        #             y_1 = y_patch_1 + patch_y_start
        #             x_1 = x_patch_1 + patch_x_start
        #             s = self.makeStroke(brush_size, x_1, y_1, referenceImage)
        #             S.append(s)

        for x in range(0, imageWidth, grid):
            for y in range(0, imageHeight, grid):
                patch = D[y:y+grid, x:x+grid]
                areaError = patch.sum()/(grid*grid)
                if areaError > self.style.threshold:
                    y1, x1 = np.unravel_index(np.argmax(patch), patch.shape)
                    s = self.makeSplineStroke(R, y1+y, x1+x, referenceImage)
                    S[tuple(s)] = tuple(self.stroke_color)

        keys = list(S.keys())
        random.shuffle(keys)
        for s in keys:
            alpha = self.draw_spline(list(s), R) 
            self.canvas = self.blend(self.canvas, alpha, np.array(S[s]))  # (H, W, 3)
        return S


    def paint(self):
        """
        :param img_path: path to the source image
        :param brushes: list with number of different brush radius
        :return:
        """
        img_path = self.style.img_path
        sourceImg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) #rgb, H*W*C

        # a new constant color image
        # now the constant color is 0/black
        self.canvas = np.zeros(sourceImg.shape) 

        # paint the canvas from the biggest brush to the smallest brush
        brushes = self.style.brush_sizes
        brush_sizes = sorted(brushes, reverse=True)
        for R in brush_sizes:
            sigma = int(self.style.f_sigma * R)
            # apply Gaussian blur
            # the size of kernel must be odd
            if sigma%2 == 0:
                sigma += 1
            referenceImage = cv2.GaussianBlur(sourceImg, (sigma, sigma), sigma)
            # referenceImage_bgr = cv2.cvtColor(referenceImage, cv2.COLOR_RGB2BGR)
            # out_path = os.path.join(self.style.out_dir, f'referenceImage{R}.jpeg')
            # cv2.imwrite(out_path, referenceImage_bgr)
            # break
            # paint a layer
            self.paintLayer(self.canvas, referenceImage, R)
            name = os.path.basename(self.style.img_path)[:-4]
            out_path = os.path.join(self.style.out_dir, f'{name}_level_{R}.jpeg')
            canvas_bgr = cv2.cvtColor(cv2.convertScaleAbs(self.canvas), cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, canvas_bgr)
            print(canvas_bgr)
            break
        

if __name__ == '__main__':
    style = Style()
    Painter(style=style).paint()
    
    
