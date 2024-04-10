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
<em>image_caption</em>

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
### Following are the optimization plots to compare-
| <img src="/state-of-the-art_methods/gabor-filters/comb_.png" alt="gab" width="400" height="275"> |
|:--:| 
| *Optimization plot for Gabor 7x7 filter* |

| <img src="//state-of-the-art_methods/laplacian-2/comb.png" alt="lap" width="400" height="275"> |
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
| *Optimization plot for FFT based approach* |

| <img src="/state-of-the-art_methods/vgg16/vgg16_base.png" alt="vggbase" width="400" height="275"> |
|:--:| 
| *Optimization plot for VGG16 without having any kernel* |

| <img src="/state-of-the-art_methods/vgg16/vgg16_3x3_kernel.png" alt="vgg33" width="400" height="275"> |
|:--:| 
| *Optimization plot for VGG16 with a 3x3 custom kernel* |

optimization plot using Lappacian filter-
<img src="" alt="lap" width="400" height="100">
<img src="/fft-based-approach/fft_denoised.png" alt="fft_denoised" width="400" height="100">
<img src="/fft-based-approach/fft_denoised.png" alt="fft_denoised" width="400" height="100">
<img src="/fft-based-approach/fft_denoised.png" alt="fft_denoised" width="400" height="100">
<img src="/fft-based-approach/fft_denoised.png" alt="fft_denoised" width="400" height="100">

| ![space-1.jpg](http://www.storywarren.com/wp-content/uploads/2016/09/space-1.jpg) | 
|:--:| 
| *Space* |
