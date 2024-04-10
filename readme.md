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
