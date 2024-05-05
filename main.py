import cv2
import numpy as np
import random
import scipy.interpolate as si
import os
from style import Style

class Painter():
    def __init__(self, style):
        self.style = style

    def calculate_difference(self, canvas, referenceImage):
        return np.sqrt(np.sum((canvas - referenceImage)**2, axis=-1))
    
    def makeSplineStroke(self, R, y0, x0, referenceImage):
        self.stroke_color = referenceImage[y0, x0]
        K = [(y0, x0)] # a new stroke with radius R
        x, y = x0, y0
        lastDx, lastDy = 0, 0
        H, W, _ = referenceImage.shape

        for i in range(1, self.style.max_stroke_len+1):
            if i > self.style.min_stroke_len and (np.sum(abs(referenceImage[y, x] - self.canvas[y, x])) < 
                                                  np.sum(abs(referenceImage[y, x] - self.stroke_color))):
                return K
        
            # get gradients
            gx, gy = self.grad_x[y][x], self.grad_y[y][x]
            gMag = (gx**2 + gy**2)**0.5

            #detect vanishing gradient
            if gMag == 0:
                return K
            
            # get unit vector of gradient
            gx = gx/gMag
            gy = gy/gMag
    
            # normal vector
            dx, dy = -gy, gx

            # if necessary, reverse direction
            if lastDx * dx + lastDy * dy < 0:
                dx, dy = -dx, -dy
            
            # filter the stroke direction
            fc = self.style.curvature_filter
            dx =  fc*dx + (1-fc)*lastDx
            dy =  fc*dy + (1-fc)*lastDy
            dx = dx/((dx**2+dy**2)**0.5)
            dy = dy/((dx**2+dy**2)**0.5)
            x, y = int(x+R*dx), int(y+R*dy)

            if x<0 or x>W-1 or y<0 or y>H-1:
                return K
            
            lastDx, lastDy = dx, dy

            K.append((y, x))

        return K
    


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

        grid = max(int(self.style.grid_size * R), 1) # grid!=0
        imageHeight, imageWidth, _ = canvas.shape

        # convert to luminance 
        weights = np.array([0.30, 0.59, 0.11], dtype=np.float32)
        luminance = np.clip(np.sum(referenceImage.astype(np.float32) * weights, axis=2), 0, 255)
        self.grad_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)    
        self.grad_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3) 
        
        for x in range(0, imageWidth, grid):
            for y in range(0, imageHeight, grid):
                patch = D[y:y+grid, x:x+grid]
                areaError = patch.sum()/(grid*grid)
                if areaError > self.style.threshold:
                    y1, x1 = np.unravel_index(np.argmax(patch), patch.shape)
                    s = self.makeSplineStroke(R, y1+y, x1+x, referenceImage)
                    S[tuple(s)] = tuple(self.stroke_color)

        # draw strokes in random order
        self.drawSplineStrokes(S, R, imageHeight, imageWidth)

        return S
    
    
    def drawSplineStrokes(self, S, R, imageHeight, imageWidth):
        keys = list(S.keys())
        random.shuffle(keys)
        for s in keys:
            for point in s:
                for i in range(point[0]-R,point[0]+R):
                    for j in range(point[1]-R,point[1]+R):
                        if np.sqrt((i - point[0])**2 + (j - point[1])**2) <= R:
                            if i>=0 and i<imageHeight and j>=0 and j<imageWidth:
                                self.canvas[i, j] = S[s]


    def paint(self):
        """
        :param img_path: path to the source image
        :param brushes: list with number of different brush radius
        :return:
        """
        img_path = self.style.img_path
        # sourceImg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) #rgb, H*W*C
        sourceImg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)/255.0 #rgb, H*W*C

        # a new constant color image
        # now the constant color is 0/black
        self.canvas = np.zeros(sourceImg.shape) 

        # paint the canvas from the biggest brush to the smallest brush
        brushes = self.style.brush_sizes
        brush_sizes = sorted(brushes, reverse=True)
        for R in brush_sizes:
            sigma = int(self.style.f_sigma * R)
            # apply Gaussian blur
            kernel_size = int(np.ceil(sigma)*6+1)
            referenceImage = cv2.GaussianBlur(sourceImg, (kernel_size, kernel_size), sigma)
            self.paintLayer(self.canvas, referenceImage, R)
            name = os.path.basename(self.style.img_path)[:-4]
            out_path = os.path.join(self.style.out_dir, f'{name}_level_{R}.jpeg')
            # canvas_bgr = cv2.cvtColor(cv2.convertScaleAbs(self.canvas), cv2.COLOR_RGB2BGR)
            canvas_bgr = cv2.cvtColor(cv2.convertScaleAbs(self.canvas*255), cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, canvas_bgr)
            print(f'Finish drawing layer level {R}.')
            # break
        

if __name__ == '__main__':
    style = Style()
    Painter(style=style).paint()
    
    
