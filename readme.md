# Painterly Rendering with Curved Brush Strokes of Multiple Sizes

Python implementation of Aaron Hertzmann's Painterly Rendering with Curved Brush Strokes of Multiple Sizes (SIGGRAPH 98) for UIUC CS445 24SP final project.

## References
Aaron Hertzmann. Painterly Rendering with Curved Brush Strokes of Multiple Sizes. Proc. SIGGRAPH 1998.
[link to the project](https://mrl.cs.nyu.edu/publications/painterly98/)
Thanks to Manuel's python implementation which can be found at [Manuel Rodriguez Ladron de Guevara's github](https://github.com/manuelladron/painterPython)
 

## Install
```
git clone https://github.com/pikapi25/painterly-rendering-python.git

```
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
               [adjust_bg] [--out_dir]
```
By default, it will generate a "results" folder within the project directory, which will contain result images for all styles of the image "tomato83.jpg". 


## Painting techniques
1. varying the brush size   
   $R_i = [R_1, R_2, ..., R_n]$
   for each layer $R_i$,   
   a. create a reference image by blurring the source image. Blurring: Gaussian kernel of std deviation: $f_{\sigma}R_i$  
   b. use a subroutine to paint each layer with $R_i$ based on the reference image. Areas that match the source image color within a threshold ($T$) are left unchanged.  

2. creating curved brush strokes  

3. paint all strokes in S on the canvas in random order

## Result
experiment with the poster of〈BORDER: DAY ONE〉by ENHYPEN
original image: 

<img src= https://github.com/pikapi25/painterly-rendering-python/blob/main/images/enhypen.jpg width=40% />

drawings at different layers in Expressionist style 

![image](https://github.com/pikapi25/painterly-rendering-python/blob/main/images/result1.png)

different styles 

![image](https://github.com/pikapi25/painterly-rendering-python/blob/main/images/result2.png)