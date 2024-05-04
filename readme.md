# Painterly Rendering with Curved Brush Strokes of Multiple Sizes

 Python implementation of Aaron Hertzmann's Painterly Rendering with Curved Brush Strokes of Multiple Sizes (SIGGRAPH 98) for UIUC CS445 24SP final project.

 ## References
 Aaron Hertzmann. Painterly Rendering with Curved Brush Strokes of Multiple Sizes. Proc. SIGGRAPH 1998.
 [link to the project](https://mrl.cs.nyu.edu/publications/painterly98/)
 Unofficial python implementation can be found at [Manuel Rodriguez Ladron de Guevara's github](https://github.com/manuelladron/painterPython)
 

 ## Introduction
 input of the main painting function:
    source image
    a list of size of brush sizes
 ### Painting techniques
 1. varying the brush size: 
    $R_i = [R_1, R_2, ..., R_n]$
    for each layer $R_i$, 
    a. create a reference image by blurring the source image. Blurring: Gaussian kernel of std deviation: $f_{\sigma}R_i$
    b. use a subroutine (loop?) to paint each layer with $R_i$ based on the reference image. Areas that match the source image color within a threshold ($T$) are left unchanged.

    ![](/images/pseudocode1.png "A pseudocode summary of the algorithm")
    ![](/images/pseudocode2.png "A pseudocode summary of the algorithm")

    formula for color difference:
    $|(r_1,g_1,b_1)-(r_2,g_2,b_2)| = \sqrt{(r_1-r_2)^2+(g_1-g_2)^2+(b_1-b_2)^2}$

 2. creating curved brush strokes