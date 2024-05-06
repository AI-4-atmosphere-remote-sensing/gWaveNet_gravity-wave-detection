# This repo provides additional supplments for the submitted work.

## All source codes are in the "source code" dir.

```gWaveNet_gravity-wave-detection/source code```

## state-of-the-art comparision
### 1 -Gabor-
#### Filters used 
```orientation of angels are: [0, 30, 60, 120, 150]```

<img src="/state-of-the-art_methods/gabor-filters/gabor_filters.png" alt="Gabor Filters" width="600" height="150">

#### Results after Gabor Filters used
<img src="/state-of-the-art_methods/gabor-filters/gabor_filters_in_action.png" alt="Gabor Filters" width="500" height="500">

#### As we can see, applying the gabor filters did not result in gravity wave detection.

### 2 -Sobel Filters used PRIOR training-
#### Filters are as follows
HORIZONTAL
 ```python
 [ -1,  0,  1],
 [ -2,  0,  2],
 [ -1,  0,  1]
 ```
VERTICAL
```python
[ -1, -2, -1],
[  0,  0,  0],
[  1,  2,  1]
```

### 3 -Sobel Filters used DURING the training-
#### Filters (same as above)


### 4 -Laplacian Filter used PRIOR training-
#### Filters are as follows
 ```python
 [ 0,  0,  1,  1,  1,  0,  0],
 [ 0,  1,  3,  3,  3,  1,  0],
 [ 1,  3,  0, -7,  0,  3,  1],
 [ 1,  3, -7,-24, -7,  3,  1],
 [ 1,  3,  0, -7,  0,  3,  1],
 [ 0,  1,  3,  3,  3,  1,  0],
 [ 0,  0,  1,  1,  1,  0,  0]
 ```

### 5 -Laplacian Filter used DURING the training-
#### Filters (same as above)


#### We applied FFT denoising technique which shows potential loss in our object of interest (GW) as follows

<img src="/fft-based-approach/fft_denoised.png" alt="fft_denoised" width="400" height="100">


## Ablation study
### 1 -Custom Kernel-
#### Our proposed Kernel [a 7x7 example]
```python
[1, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 1, 0],
[1, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 1, 0],
[1, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 1, 0],
[1, 0, 1, 0, 1, 0, 1]
```
### Following are the optimization plots for State-of-the-art techniques-
| <img src="/state-of-the-art_methods/gabor-filters/comb_.png" alt="gab" width="400" height="275"> |
|:--:| 
| *Optimization plot for Gabor 7x7 filter* |

| <img src="/state-of-the-art_methods/laplacian-1/comb_.png" alt="lap" width="400" height="275"> |
|:--:| 
| *Optimization plot for Laplacian 7x7 filter* |

| <img src="/state-of-the-art_methods/sobel-filters_1/comb_.png" alt="sob" width="400" height="275"> |
|:--:| 
| *Optimization plot for Sobel 3x3 filter* |

| <img src="/state-of-the-art_methods/ViT/vit.png" alt="vit" width="400" height="275"> |
|:--:| 
| *Optimization plot for Vision Transformer based approach* |

| <img src="/fft-based-approach/fft-denoised-data-trainable-kernel/fft_based.png" alt="fft" width="400" height="275"> |
|:--:| 
| *Optimization plot for FFT based approach using a 7x7 kernel* |

| <img src="/state-of-the-art_methods/vgg16/vgg16_base.png" alt="vggbase" width="400" height="275"> |
|:--:| 
| *Optimization plot for VGG16 without having any kernel* |

| <img src="/state-of-the-art_methods/vgg16/vgg16_3x3_kernel.png" alt="vgg33" width="400" height="275"> |
|:--:| 
| *Optimization plot for VGG16 with a 3x3 custom kernel* |

### Following are the optimization plots from our proposed approach (ablation studies)-
| <img src="/ablation-study/custom-kernel-applied-trainable/opt-5x5-t.png" alt="5x5" width="400" height="275"> |
|:--:| 
| *Optimization plot using a proposed 5x5 checkerboard kernel* |

| <img src="/ablation-study/custom-kernel-applied-trainable/opt-7x7.png" alt="7x7" width="400" height="275"> |
|:--:| 
| *Optimization plot using a proposed 7x7 checkerboard kernel* |

| <img src="/ablation-study/custom-kernel-applied-trainable/opt-9x9.png" alt="9x9" width="400" height="275"> |
|:--:| 
| *Optimization plot using a proposed 9x9 checkerboard kernel* |

| <img src="/ablation-study/custom-kernel-applied-trainable/opt-multi_k.png" alt="mul" width="400" height="275"> |
|:--:| 
| *Optimization plot using multiplw 7x7 kernels* |
