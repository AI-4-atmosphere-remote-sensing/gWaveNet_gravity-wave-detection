# This repo provides additional supplments for the submitted work.

## state-of-the-art comparision
### 1 -Gabor-
#### Filters used 
```orientation of angels are: [0, 30, 60, 120, 150]```

<img src="/state-of-the-art_methods/gabor-filters/gabor_filters.png" alt="Gabor Filters" width="600" height="150">

#### Optimization 

<img src="/state-of-the-art_methods/gabor-filters/comb_.png" alt="Gabor Filters" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/gabor-filters/```

### 2 -Sobel Filters used PRIOR training
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

#### Optimization 

<img src="/state-of-the-art_methods/sobel-filters_1/comb_.png" alt="son-prr" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/sobel-filters_1/```

### 3 -Sobel Filters used DURING the training
#### Filters (same as above)
#### Optimization 

<img src="/state-of-the-art_methods/sobel-filters-2/comb_.png" alt="sob-dur" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/sobel-filters-2/```

### 4 -Laplacian Filter used PRIOR training
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
#### Optimization 

<img src="/state-of-the-art_methods/laplacian-1/comb_.png" alt="lap-prr" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/laplacian-1/```

### 5 -Laplacian Filter used DURING the training
#### Filters (same as above)
#### Optimization 

<img src="/state-of-the-art_methods/laplacian-2/comb.png" alt="lap-dur" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/laplacian-2/```

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
#### Optimization results for a vanilla model (with no custom kernel applied) - TEST_1

<img src="/ablation_study/no-custom-kernel-applied/comb_1.png" alt="v1" width="450" height="300">

#### Optimization results for a vanilla model (with no custom kernel applied) - TEST_2

<img src="/ablation_study/no-custom-kernel-applied/comb_2.png" alt="v2" width="450" height="300">


#### Optimization results using the kernel above with trainable layers

<img src="/ablation_study/custom-kernel-applied-trainable/comb_.png" alt="ck" width="450" height="300">


#### Optimization results using the kernel above with non-trainable layers

<img src="/ablation_study/custom-kernel-applied-non-trainable/comb_.png" alt="ck" width="450" height="300">


