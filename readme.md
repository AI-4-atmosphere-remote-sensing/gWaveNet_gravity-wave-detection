# This repo provides additional supplments for the submitted work.

## state-of-the-art comparision
### 1 -Gabor-
#### Filters used 
```orientation of angels are: [0, 30, 60, 120, 150]```

<img src="/state-of-the-art_methods/gabor-filters/gabor_filters.png" alt="Gabor Filters" width="600" height="150">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/gabor-filters/```

#### Results after Gabor Filters used
<img src="/state-of-the-art_methods/gabor-filters/g1.png" alt="Gabor Filters" width="400" height="100">
<img src="/state-of-the-art_methods/gabor-filters/g2.png" alt="Gabor Filters" width="400" height="100">
<img src="/state-of-the-art_methods/gabor-filters/g3.png" alt="Gabor Filters" width="400" height="100">

#### As we can see, applying the gabor filters did not result in gravity wave detection.

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



#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/sobel-filters_1/```

### 3 -Sobel Filters used DURING the training
#### Filters (same as above)


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


#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/laplacian-1/```

### 5 -Laplacian Filter used DURING the training
#### Filters (same as above)


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


