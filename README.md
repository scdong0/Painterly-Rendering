# Painterly Rendering with Curved Brush Strokes of Multiple Sizes

Python implementation of Aaron Hertzmann's Painterly Rendering with Curved Brush Strokes of Multiple Sizes (SIGGRAPH 98) for UIUC CS445 24SP final project.

## References
Aaron Hertzmann. Painterly Rendering with Curved Brush Strokes of Multiple Sizes. Proc. SIGGRAPH 1998.
[link to the project](https://mrl.cs.nyu.edu/publications/painterly98/)
Thanks to Manuel's python implementation which can be found at [Manuel Rodriguez Ladron de Guevara's github](https://github.com/manuelladron/painterPython)
 

## Installation
```
conda install opencv
conda install numpy
conda install scipy
```

## To run
```
python main.py [--img_path] [--brushes] [--f_sigma] 
               [--maxLength] [--minLength] [--threshold] 
               [--grid_size] [--curvature_filter] 
               [--alpha] [--hsvjit] [--rgbjit] 
               [--adjust_bg] [--out_dir]
```
By default, it will generate a "results" folder within the project directory, which will contain result images for all styles of the image "tomato83.jpg". 

