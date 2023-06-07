# This repo provides additional supplments for the submitted work.

## state-of-the-art comparision
### 1 -Gabor-
#### Filters used 

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

<img src="/state-of-the-art_methods/sobel-filters_1/comb_.png" alt="Gabor Filters" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/sobel-filters_1/```

### 3 -Sobel Filters used DURING the training
#### Filters (same as above)
#### Optimization 

<img src="/state-of-the-art_methods/sobel-filters-2/comb_.png" alt="Gabor Filters" width="450" height="300">

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

<img src="/state-of-the-art_methods/laplacian-1/comb_.png" alt="Gabor Filters" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/laplacian-1/```

### 5 -Laplacian Filter used DURING the training
#### Filters (same as above)
#### Optimization 

<img src="/state-of-the-art_methods/laplacian-2/comb.png" alt="Gabor Filters" width="450" height="300">

#### Code (model_gabor.py) including filters and output (out.out) are in the:

```gWaveNet_gravity-wave-detection/state-of-the-art_methods/laplacian-2/```
