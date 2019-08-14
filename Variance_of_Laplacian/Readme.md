# Blur Detection using Variance of Laplacian

[Here](http://optica.csic.es/papers/icpr2k.pdf) is the original paper presented in 2000, ICPR by Pech-Pacheco et. al. We simply take a single channel of a grayscale image and colvolve it with the following 3X3 kernel.

```
[ 0,  1,  0]
[ 1, -4,  1]
[ 0,  1,  0]
```

and then take the variance of the response. If the variance falls below a pre-defined threshold, then the images is considered blurry, otherwise the image is not blurry.

### Disadvantage :
 - [ ] There is no universal threshold value which can help filter blurry images. The threshold value may change for different types of images.
