import argparse
import cv2
from networkx import difference
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='images/tomato83.jpg')
parser.add_argument('--brushes', type=list, default=[8,4,2])
parser.add_argument('--f_sigma', type=float, default=1)
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--len_stroke', type=float, default=1)
parser.add_argument('--maxLength', type=int, default=16)
parser.add_argument('--minLength', type=int, default=4)
# parser.add_argument('--resize', type=list, default=None)
# parser.add_argument('--blur_fac', type=float, default=.5)
# parser.add_argument('--grid_fac', type=float, default=1)

parser.add_argument('--filter_fac', type=float, default=1)
args = parser.parse_args()

class Painter():
    """
    """
    def __init__(self, args):
        self.args = args
        self.img_path = args.img_path
        self.f_sigma = args.f_sigma
        self.T = args.threshold
        self.len_stroke = args.len_stroke
        self.maxLength = args.maxLength
        self.minLength = args.minLength
        self.filter_fac = args.filter_fac
        self.paint(self.img_path, self.brushs)

    def calculate_difference(self, canvas, referenceImage):
        r_diff = canvas[:, :, 0]-referenceImage[:, :, 0]
        g_diff = canvas[:, :, 1]-referenceImage[:, :, 1]
        b_diff = canvas[:, :, 2]-referenceImage[:, :, 2]
        return np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)
    
    def makeStroke(self, brush_size, x, y, referenceImage):
        """
        Given starting position [row (idx_x), col (idx_y)] and a stroke max length:
        (1) Iterate over max length pixels
        (2) Calculate gradient (direction, normal, magnitude) at x, y position
        (3) Add dx, dy to list of control pts and update idx_x, idx_y
        :param r: stroke thickness
        :param idx_row: starting position y (row)
        :param idx_col: starting position x (col)
        :param referenceImage:
        :return: list of control points
        """
        ref_color = referenceImage[y, x, :] # rgb
        self.stroke_color = ref_color

        canvas_color = self.canvas[y, x, :]
        start_pts = [(y, x)] # Stroke starts here
        last_dy, last_dx = 0,0

        length = int(brush_size * self.len_stroke)
        imageHeight, imageWidth, channel = referenceImage.shape

        for i in range(self.maxLength):
            # Off boundaries check
            x = max(min(x, imageWidth-1), 0)
            y = max(min(y, imageHeight-1), 0)

            # canvas at this point is already properly colored (diff bt canvas and ref is less than ref and stroke)
            if (i > self.minLength) and (np.sqrt(ref_color**2 - canvas_color**2).mean() < np.sqrt(ref_color**2 -
                                                                                           self.stroke_color**2).mean()):
                return start_pts

            # (2) Calculate gradients and its magnitude
            gx, gy = np.sum(self.grad_x[y, x]), np.sum(self.grad_y[y, x])
            g_mag = np.sqrt(gx**2 + gy**2)

            # Compute normal
            dx, dy = -gy, gx

            # if gradient is small, return control points
            if length*g_mag < 1:
                return start_pts

            # if necessary, reverse direction
            if i > 0 and (last_dx * dx + last_dy * dy) < 0:
                dx, dy = -dx, -dy

            # filter the stroke direction
            dx = (1-self.filter_fac)*last_dx+self.filter_fac*dx
            dy = (1-self.filter_fac)*last_dy+self.filter_fac*dy

            # Compute new magnitude
            g_mag = np.sqrt(dx**2 + dy**2)

            # New points - "The distance between ctrl points is equal to the brush radius"
            x = int(x + length * dx/g_mag)
            y = int(y + length * dy/g_mag)

            start_pts.append((y, x))  # [H,W]

            # Update last_dx, last_dy
            last_dx = dx
            last_dy = dy

        return start_pts

    def paintLayer(self, canvas, referenceImage, brush_size):
        """
        :param canvas: numpy (H*W*C)
        :param referenceImage: reference image for this layer (H*W*C)
        :param brush_size: brush radius used in this layer (int)
        :return: painted_layer (H*W*C)
        """
        S = [] #empty strokes

        # create a pointwise difference image
        D = self.calculate_difference(canvas, referenceImage)

        grid = max(int(self.f_sigma * R), 1) #grid!=0
        half = grid/2
        imageHeight, imageWidth, channel = canvas.shape

        # convert to luminance 
        referenceImage_float = referenceImage.astype(np.float32)
        R = referenceImage_float[:, :, 0]
        G = referenceImage_float[:, :, 1]
        B = referenceImage_float[:, :, 2]
        luminance = 0.30 * R + 0.59 * G + 0.11 * B #luminance in float32
        # luminance = np.clip(Y, 0, 255).astype(np.uint8)
        self.grad_x = cv2.Sobel(luminance, cv2.CV_64F, 1, 0, ksize=3)    
        self.grad_y = cv2.Sobel(luminance, cv2.CV_64F, 0, 1, ksize=3) 

        for x in range(half, imageWidth-half+1, grid):
            for y in range(half, imageHeight-half+1, grid):
                # sum the error near (x,y)
                patch_x_start = max(0, x-half)
                patch_x_end = min(x+half, imageWidth)
                patch_y_start = max(0, y-half)
                patch_y_end = min(y+half, imageHeight)
                # patch_ref = referenceImage[patch_y_start:patch_y_end, patch_x_start:patch_x_end, :]
                # patch_canvas = canvas[patch_y_start:patch_y_end, patch_x_start:patch_x_end, :]

                D_patch = D[patch_y_start:patch_y_end, patch_x_start:patch_x_end, :]
                D_patch_mean = np.mean(D_patch, axis=2) # D_patch_mean (H*W)
                areaError = np.mean(D_patch_mean)

                if(areaError<self.T):
                    # find the largest error point
                    y_patch_1, x_patch_1 = np.unravel_index(np.argmax(D_patch_mean), D_patch_mean.shape)
                    y_1 = y_patch_1 + patch_y_start
                    x_1 = x_patch_1 + patch_x_start
                    s = self.makeStroke(brush_size, x_1, y_1, referenceImage)
                    S.append(s)
        return S


    def paint(self, img_path, brushes):
        """
        :param img_path: path to the source image
        :param brushes: list with number of different brush radius
        :return:
        """
        
        sourceImg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) #rgb, H*W*C

        # a new constant color image
        # now the constant color is 0/black
        self.canvas = np.zeros(sourceImg.shape) 

        # paint the canvas from the biggest brush to the smallest brush
        brush_sizes = sorted(brushes, reverse=True)
        for R in brush_sizes:
            sigma = int(self.f_siga * R)
            # apply Gaussian blur
            referenceImage = cv2.GaussianBlur(sourceImg, (sigma, sigma), sigma)
            # paint a layer
            self.paintLayer(self.canvas, referenceImage, brush_sizes)
        
