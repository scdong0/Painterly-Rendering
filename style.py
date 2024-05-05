import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='images/tomato83.jpg')
parser.add_argument('--brushes', type=list, default=[8,4,2])
parser.add_argument('--f_sigma', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=0.05)
parser.add_argument('--maxLength', type=int, default=16)
parser.add_argument('--minLength', type=int, default=4)
parser.add_argument('--grid_size', type=float, default=1)
parser.add_argument('--curvature_filter', type=float, default=1)
parser.add_argument('--out_dir', type=str, default='.')
args = parser.parse_args()

class Style:
    def __init__(self):
        self.img_path = args.img_path
        self.brush_sizes = args.brushes
        self.f_sigma = args.f_sigma
        self.threshold = args.threshold
        self.max_stroke_len = args.maxLength
        self.min_stroke_len = args.minLength
        self.grid_size = args.grid_size
        self.curvature_filter = args.curvature_filter
        self.out_dir = args.out_dir
       
        
        