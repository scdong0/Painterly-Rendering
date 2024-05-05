import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='images/lizard1.jpg')
parser.add_argument('--brushes', type=list, default=[8,4,2])
parser.add_argument('--f_sigma', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=30)
parser.add_argument('--maxLength', type=int, default=16)
parser.add_argument('--minLength', type=int, default=4)
parser.add_argument('--grid_size', type=float, default=1)
parser.add_argument('--curvature_filter', type=float, default=1)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--out_dir', type=str, default='results')
args = parser.parse_args()

# defalt is impressionist, customize py passing arguments
class Style:
    def __init__(self):
        self.name = "Default"
        self.img_path = args.img_path
        self.brush_sizes = args.brushes
        self.f_sigma = args.f_sigma
        self.threshold = args.threshold
        self.max_stroke_len = args.maxLength
        self.min_stroke_len = args.minLength
        self.grid_size = args.grid_size
        self.curvature_filter = args.curvature_filter
        self.alpha = args.alpha
        self.out_dir = args.out_dir
       
        
class Impressionist(Style):
    def __init__(self):
        super().__init__()
        self.name = "Impressionist"
        self.threshold = 30
        self.brush_sizes = [8, 4, 2]
        self.curvature_filter = 1.
        self.f_sigma = .5
        self.min_stroke_len = 4
        self.max_stroke_len = 16
        self.grid_size = 1.
        self.alpha = 1.

class Expressionist(Style):
    def __init__(self):
        super().__init__()
        self.name = "Expressionist"
        self.threshold = 20
        self.brush_sizes = [8, 4, 2]
        self.curvature_filter = .25
        self.f_sigma = .5
        self.min_stroke_len = 10
        self.max_stroke_len = 16
        self.grid_size = 1.
        self.alpha = .7


class ColoristWash(Style):
    def __init__(self):
        super().__init__()
        self.name = "ColoristWash"
        self.threshold = 75
        self.brush_sizes = [8, 4, 2]
        self.curvature_filter = 1.
        self.f_sigma = .5
        self.min_stroke_len = 4
        self.max_stroke_len = 16
        self.grid_size = 1.
        self.alpha = .5


class Pointillist(Style):
    def __init__(self):
        super().__init__()
        self.name = "Pointillist"
        self.threshold = 50
        self.brush_sizes = [4, 2]
        self.curvature_filter = 1.
        self.f_sigma = .5
        self.min_stroke_len = 0
        self.max_stroke_len = 0
        self.grid_size = 0.5
        self.alpha = 1.

class Cartoon(Style):
    def __init__(self):
        super().__init__()
        self.name = "Cartoon"
        self.threshold = 60
        self.brush_sizes = [20, 10, 6, 1]
        self.curvature_filter = 1.
        self.f_sigma = .5
        self.min_stroke_len = 2
        self.max_stroke_len = 8
        self.grid_size = 1.
        self.alpha = 1.
       
class Abstract(Style):
    def __init__(self):
        super().__init__()
        self.name = "Abstract"
        self.threshold = 75
        self.brush_sizes = [32,16, 8, 4,2]
        self.curvature_filter = .5
        self.f_sigma = .5
        self.min_stroke_len = 8
        self.max_stroke_len = 48
        self.grid_size = 5.
        self.alpha = .8