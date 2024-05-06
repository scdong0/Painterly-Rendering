import cv2
import numpy as np
import random
import scipy.interpolate as si
import os
from style import Style, Impressionist, Expressionist, ColoristWash, Pointillist, Cartoon, Abstract
import colorsys

class Painter():
    def __init__(self, style):
        self.style = style

    def calculate_difference(self, canvas, refImage):
        return np.sqrt(np.sum((canvas - refImage)**2, axis=-1))
    
    def makeSplineStroke(self, R, y0, x0, refImage):
        self.stroke_color = refImage[y0, x0]
        K = [(y0, x0)]      # a new stroke with radius R
        x, y = x0, y0
        lastDx, lastDy = 0, 0
        H, W, _ = refImage.shape

        for i in range(1, self.style.max_stroke_len+1):
            if i > self.style.min_stroke_len and (np.sum(abs(refImage[y, x] - self.canvas[y, x])) < 
                                                  np.sum(abs(refImage[y, x] - self.stroke_color))):
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
    


    def paintLayer(self, canvas, refImage, R):
        """
        :param canvas: numpy (H*W*C)
        :param refImage: reference image for this layer (H*W*C)
        :param brush_size: brush radius used in this layer (int)
        :return: painted_layer (H*W*C)
        """
        S = {} #empty strokes

        # create a pointwise difference image
        D = self.calculate_difference(canvas, refImage)

        grid = max(int(self.style.grid_size * R), 1) # grid!=0
        imageHeight, imageWidth, _ = canvas.shape

        # convert to luminance 
        weights = np.array([0.30, 0.59, 0.11], dtype=np.float32)
        luminance = np.clip(np.sum(refImage[:,:,:3].astype(np.float32) * weights, axis=2), 0, 255)
        self.grad_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)    
        self.grad_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3) 
        
        for x in range(0, imageWidth, grid):
            for y in range(0, imageHeight, grid):
                patch = D[y:y+grid, x:x+grid]
                areaError = patch.sum()/(grid*grid)
                if areaError > self.style.threshold:
                    y1, x1 = np.unravel_index(np.argmax(patch), patch.shape)
                    s = self.makeSplineStroke(R, y1+y, x1+x, refImage)
                    
                    # add color jitters
                    hjit, sjit, vjit = self.style.hsvjit
                    if hjit != 0 or sjit != 0 or vjit != 0:
                        self.stroke_color = self.adjustColor(self.stroke_color, 1, hjit, 1, sjit, 1, vjit)
                    rjit, gjit, bjit = self.style.rgbjit
                    if rjit != 0 or gjit != 0 or bjit != 0:
                        self.stroke_color = self.addRGBjit(self.stroke_color, rjit, gjit, bjit)

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
                                r, g, b = S[s]
                                alphaB = self.alphaB[i, j]
                                alpha = self.style.alpha
                                alphaF = alpha + alphaB*(1-alpha)
                                r_out = min(int((r*alpha+self.canvas[i][j][0]*alphaB*(1-alpha))/alphaF), 255)
                                g_out = min(int((g*alpha+self.canvas[i][j][1]*alphaB*(1-alpha))/alphaF), 255)
                                b_out = min(int((b*alpha+self.canvas[i][j][2]*alphaB*(1-alpha))/alphaF), 255)
                                
                                self.alphaB[i, j] = alphaF
                                self.canvas[i, j] = r_out, g_out, b_out


    def addRGBjit(self, color, rjit, gjit, bjit):
        r, g, b = color

        r += int(rjit * (random.random() - 0.5) * 255)
        g += int(gjit * (random.random() - 0.5) * 255)
        b += int(bjit * (random.random() - 0.5) * 255)

        r = min(max(r, 0), 255)
        g = min(max(g, 0), 255)
        b = min(max(b, 0), 255)

        return r, g, b

    
    def adjustColor(self, color, hfac, hjit, sfac, sjit, bfac, bjit):
        # Extracting RGB components
        r, g, b = color

        # Converting to HSB
        f = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        # Adjusting HSB
        f = list(f)
        f[0] = (hfac * (f[0] + hjit * (random.random() - 0.5))) % 1
        f[1] = max(min(sfac * (f[1] + sjit * (random.random() - 0.5)), 1), 0)
        f[2] = max(min(bfac * (f[2] + bjit * (random.random() - 0.5)), 1), 0)

        # Converting back to RGB
        new_color = colorsys.hsv_to_rgb(f[0], f[1], f[2])
        new_color = tuple(int(c * 255) for c in new_color)
        return new_color
    

    def aveColor(self, img):
        r = np.mean(img[:, :, 0])
        g = np.mean(img[:, :, 1])
        b = np.mean(img[:, :, 2])
        r = min(int(r), 255)
        g = min(int(g), 255)
        b = min(int(b), 255)
        return r, g, b
    

    def paint(self):
        """
        :param img_path: path to the source image
        :param brushes: list with number of different brush radius
        :return:
        """
        img_path = self.style.img_path
        sourceImg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) #rgb, H*W*C

        # a new constant color image
        canvasColor = self.adjustColor(self.aveColor(sourceImg),1,0,20,0,1,0)
        self.canvas = np.full(sourceImg.shape, canvasColor, dtype=np.float32)

        # paint the canvas from the biggest brush to the smallest brush
        brushes = self.style.brush_sizes
        brush_sizes = sorted(brushes, reverse=True)
        self.alphaB = np.zeros((sourceImg.shape[0], sourceImg.shape[1]))
        for R in brush_sizes:
            # apply Gaussian blur
            sigma = int(self.style.f_sigma * R)
            kernel_size = int(np.ceil(sigma)*6+1)
            refImage = cv2.GaussianBlur(sourceImg, (kernel_size, kernel_size), sigma)

            # draw
            self.paintLayer(self.canvas, refImage, R)
            img_name = os.path.basename(self.style.img_path)[:-4]
            out_dir = os.path.join(self.style.out_dir, img_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, f'{self.style.name}_level_{R}.jpg')
            canvas_bgr = cv2.cvtColor(cv2.convertScaleAbs(self.canvas), cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, canvas_bgr)
            print(f'Finish drawing {img_name} in {self.style.name} style at layer level {R}.')
        

if __name__ == '__main__':
    # create your own painting style by passing auguments
    # styles = {Style()}       # uncomment this line if you want to create your own painting style

    # some styles already created
    styles = {Impressionist(), Expressionist(), ColoristWash(), Pointillist(), Cartoon(), Abstract()}

    for style in styles:
        Painter(style).paint()
    
    
