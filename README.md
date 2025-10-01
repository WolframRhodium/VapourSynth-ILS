# VapourSynth-ILS

CUDA implementation of [Real-time Image Smoothing via Iterative Least Squares](https://dl.acm.org/doi/10.1145/3388887) for VapourSynth.


It is a global optimization based edge-preserving smoothing filter, which can avoid haloing and gradient reversal artifacts commonly found in weighted average based methods like bilateral filter and guided filter.


## Requirements

- CUDA-enabled GPU(s).

- [cuFFT library](https://developer.nvidia.com/cufft), i.e. `cufft64_*.dll` on Windows or `libcufft.so.*` on Linux.


## Parameters

```python3
ils.ILS(clip clip[, float lambda=0.5, int iteration=4, float p=0.8, float eps=0.0001, float gamma=None, float c=None, bool use_welsch=False, int device_id=0, int num_streams=2, bool use_cuda_graph=True])
```

In short, use `use_welsch=True` with `lambda`, `iterations`, `gamma` for compression artifacts removal tasks or `use_welsch=False` with `lambda`, `iterations`, `p` for detail manipulation tasks.

- `clip`

    The input clip. Must be of 32 bit float format. Only the first plane is processed.

- `lambda`

    Smoothing strength of the filter.

    Default: `0.5`

- `iteration`

    Iteration number of optimization. A larger iteration number can lead to stronger smoothing on large-amplitude details at the expense of a much higher computational cost.

    Default: `4`

- `p`

    Power norm of the penalty on gradient, which controls the sensitivity to the edges in the input image. A smaller value tends to blur smooth regions but leaving salient edges untouched. A value in 0.8 ∼ 1 may be suitable for tasks of tone and detail manipulation, which can produce results with little visible artifacts.

    Default: `0.8`

- `eps`

    Small constant to make the penalty function differentiable at the origin. A larger
leads to higher convergency speed with the risk of resulting in halo artifacts.

    Default: `0.0001`


- `gamma`, `c`

    Computed automatically.

    Default:

    - `gamma`: 0.5 * p - 1

    - `c`: p * (eps ** gamma)

- `use_welsch`:

    Whether to use the Welsch penalty function. If not, the Charbonnier penalty is used instead.

    The Welsch penalty is suitable for clip-art compression artifacts removal while the Charbonnier penalty is suitable for tone and detail manipulation.

    Default: `False`


- `device_id`

    Set GPU to be used.

    Default: `0`

- `num_streams`

    Number of CUDA streams, enables concurrent kernel execution and data transfer.

    Default: `4`

- `use_cuda_graph`

    Whether to use [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) to reduce CPU cost and kernel launch overhead.

    Default: `True`


## Compilation

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_FLAGS="--threads 0 --use_fast_math -Wno-deprecated-gpu-targets" -D CMAKE_CUDA_ARCHITECTURES="50;61-real;75-real;86"

cmake --build build --config Release
```

