#include "ils_impl.hpp"

#include <cuda_runtime.h>

#include <VSHelper.h>
#include <utility>


#define checkError(expr) do {                                                  \
    cudaError_t __err = expr;                                                  \
    if (__err != cudaSuccess) {                                                \
        abort();                                                               \
    }                                                                          \
} while(0)

#define checkCufftError(expr) do {                                             \
    cufftResult __err = expr;                                                  \
    if (__err != CUFFT_SUCCESS) {                                              \
        abort();                                                               \
    }                                                                          \
} while(0)


__global__
static void compute_denormin(
    cufftReal * __restrict__ denormin,
    const cufftComplex * __restrict__ otfFx,
    const cufftComplex * __restrict__ otfFy,
    float c, float lambda,
    int half_width, int height, int denormin_stride, int stride2
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= half_width || y >= height) {
        return ;
    }

    auto abs_sq = [](cufftComplex x) -> float { return x.x * x.x + x.y * x.y; };

    cufftReal d = abs_sq(otfFx[y * stride2 + x]) + abs_sq(otfFy[y * stride2 + x]);
    denormin[y * denormin_stride + x] = 1.0f + 0.5f * c * lambda * d;
}


template <bool use_welsch=false>
__global__
static void compute_normin2_pre(
    float * __restrict__ dst, const float * __restrict__ src,
    int width, int height, int stride,
    float c, float p, float eps, float gamma
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return ;
    }

    float u_h[2] {
        src[y * stride + x] - src[y * stride + (x == 0 ? width - 1 : x - 1)],
        src[y * stride + (x == width - 1 ? 0 : x + 1)] - src[y * stride + x]
    };
    float u_v[2] {
        src[y * stride + x] - src[(y == 0 ? height - 1 : y - 1) * stride + x],
        src[(y == height - 1 ? 0 : y + 1) * stride + x] - src[y * stride + x]
    };

    for (int i = 0; i < 2; ++i) {
        if constexpr (use_welsch) {
            u_h[i] = c * u_h[i] - 2 * u_h[i] * expf(-u_h[i] * u_h[i] / (2 * gamma * gamma));
            u_v[i] = c * u_v[i] - 2 * u_v[i] * expf(-u_v[i] * u_v[i] / (2 * gamma * gamma));
        } else {
            u_h[i] = c * u_h[i] - p * u_h[i] * powf(u_h[i] * u_h[i] + eps, gamma);
            u_v[i] = c * u_v[i] - p * u_v[i] * powf(u_v[i] * u_v[i] + eps, gamma);
        }
    }

    dst[y * stride + x] = (u_h[0] - u_h[1]) + (u_v[0] - u_v[1]);
}


__global__
static void compute_normin1(
    cufftComplex * __restrict__ normin1, float lambda,
    const cufftComplex * __restrict__ normin2,
    const float * __restrict__ denormin,
    int half_width, int height, int denormin_stride, int stride2
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= half_width || y >= height) {
        return ;
    }

    cufftComplex & normin1_v = normin1[y * stride2 + x];
    const cufftComplex & normin2_v = normin2[y * stride2 + x];
    const float & denormin_v = denormin[y * denormin_stride + x];

    normin1_v.x = (normin1_v.x + 0.5f * lambda * normin2_v.x) / denormin_v;
    normin1_v.y = (normin1_v.y + 0.5f * lambda * normin2_v.y) / denormin_v;
}


__global__
static void unscale(
    float * __restrict__ img,
    int width, int height, int stride
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return ;
    }

    img[y * stride + x] /= width * height;
}


static inline cufftResult fft_plan_2d(
    cufftHandle * plan,
    int width, int height, int istride, int ostride,
    cufftType type, int batch=1
) noexcept {

    int n[] { height, width };
    int inembed[] { 0, istride };
    int onembed[] { 0, ostride };

    auto result = cufftPlanMany(
        plan, 2, n,
        inembed, 1, height * istride,
        onembed, 1, height * ostride,
        type, batch
    );

    return result;
}


float * init_denormin(
    int width, int height, float c, float lambda
) noexcept {

    int half_width = width / 2 + 1;

    float * d_denormin;
    checkError(cudaMalloc(&d_denormin, height * half_width * sizeof(float)));

    cufftHandle rfft2d_plan;
    checkCufftError(fft_plan_2d(
        &rfft2d_plan,
        width, height, width, half_width,
        CUFFT_R2C
    ));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    checkCufftError(cufftSetStream(rfft2d_plan, stream));

    float * h_psf;
    checkError(cudaMallocHost(&h_psf, height * width * sizeof(cufftReal)));
    float * d_psf;
    checkError(cudaMalloc(&d_psf, height * width * sizeof(cufftReal)));

    cufftComplex * d_otfFx, * d_otfFy;
    checkError(cudaMalloc(&d_otfFx, height * half_width * sizeof(cufftComplex)));
    checkError(cudaMalloc(&d_otfFy, height * half_width * sizeof(cufftComplex)));

    {
        memset(h_psf, 0, height * width * sizeof(float));
        h_psf[0] = -1.0f;
        h_psf[(height - 1) * width] = 1.0f;
        checkError(cudaMemcpyAsync(
            d_psf, h_psf,
            height * width * sizeof(float),
            cudaMemcpyHostToDevice, stream
        ));
        checkCufftError(cufftExecR2C(rfft2d_plan, d_psf, d_otfFx));
        checkError(cudaStreamSynchronize(stream));
    }

    {
        h_psf[(height - 1) * width] = 0.0f;
        h_psf[width - 1] = 1.0f;
        checkError(cudaMemcpyAsync(
            d_psf, h_psf,
            height * width * sizeof(float),
            cudaMemcpyHostToDevice, stream
        ));
        checkCufftError(cufftExecR2C(rfft2d_plan, d_psf, d_otfFy));
    }

    {
        const dim3 block { 16, 8 };
        const dim3 half_grid {
            (half_width - 1) / block.x + 1,
            (height - 1) / block.y + 1
        };
        compute_denormin<<<half_grid, block, 0, stream>>>(
            d_denormin, d_otfFx, d_otfFy,
            c, lambda,
            half_width, height, half_width, half_width
        );
    }

    checkError(cudaStreamSynchronize(stream));

    checkError(cudaFree(d_otfFy));
    checkError(cudaFree(d_otfFx));
    checkError(cudaFree(d_psf));
    checkError(cudaFreeHost(h_psf));

    checkError(cudaStreamDestroy(stream));

    return d_denormin;
}


void IlsInstance::init(
    const IlsParams & param,
    int width, int height,
    bool use_cuda_graph,
    const float * d_denormin
) noexcept {

    std::exchange(this->param, param);

    this->width = width;
    this->height = height;
    this->d_denormin = d_denormin;

    int half_width = width / 2 + 1;

    checkError(cudaStreamCreate(&stream));
    checkError(cudaMalloc(&d_img, height * width * sizeof(float)));
    checkError(cudaMalloc(&d_tmp, height * width * sizeof(float)));
    checkError(cudaMalloc(&d_normin1, height * half_width * sizeof(cufftComplex)));
    checkError(cudaMalloc(&d_normin1_backup, height * half_width * sizeof(cufftComplex)));
    checkError(cudaMalloc(&d_normin2, height * half_width * sizeof(cufftComplex)));
    checkError(cudaMallocHost(&h_img, height * width * sizeof(float)));

    checkCufftError(fft_plan_2d(
        &rfft2d_plan,
        width, height,
        width, half_width,
        CUFFT_R2C
    ));
    checkCufftError(cufftSetStream(rfft2d_plan, stream));
    checkCufftError(fft_plan_2d(
        &irfft2d_plan,
        width, height,
        half_width, width,
        CUFFT_C2R
    ));
    checkCufftError(cufftSetStream(irfft2d_plan, stream));

    if (use_cuda_graph) {
        cudaGraph_t graph;

        checkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        launch();
        checkError(cudaStreamEndCapture(stream, &graph));

        checkError(cudaGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));
        checkError(cudaGraphDestroy(graph));
    } else {
        graphexec = nullptr;
    }
}


void IlsInstance::launch() noexcept {
    checkError(cudaMemcpyAsync(
        d_img, h_img, height * width * sizeof(float),
        cudaMemcpyHostToDevice, stream
    ));

    int half_width = width / 2 + 1;
    const dim3 block { 16, 8 };
    const dim3 grid {
        (width - 1) / block.x + 1,
        (height - 1) / block.y + 1,
    };
    const dim3 half_grid {
        (half_width - 1) / block.x + 1,
        (height - 1) / block.y + 1,
    };

    checkCufftError(cufftExecR2C(rfft2d_plan, d_img, d_normin1));

    for (int i = 0; i < param.iteration; ++i) {
        if (param.use_welsch) {
            compute_normin2_pre<true><<<grid, block, 0, stream>>>(
                d_tmp, d_img,
                width, height, width,
                param.c, param.p, param.eps, param.gamma
            );
        } else {
            compute_normin2_pre<false><<<grid, block, 0, stream>>>(
                d_tmp, d_img,
                width, height, width,
                param.c, param.p, param.eps, param.gamma
            );
        }

        checkCufftError(cufftExecR2C(rfft2d_plan, d_tmp, d_normin2));

        compute_normin1<<<half_grid, block, 0, stream>>>(
            d_normin1, param.lambda,
            d_normin2, d_denormin,
            half_width, height, half_width, half_width
        );

        checkError(cudaMemcpyAsync(
            d_normin1_backup, d_normin1,
            height * half_width * sizeof(cufftComplex),
            cudaMemcpyDeviceToDevice, stream
        ));
        checkCufftError(cufftExecC2R(irfft2d_plan, d_normin1, d_img));
        checkError(cudaMemcpyAsync(
            d_normin1, d_normin1_backup,
            height * half_width * sizeof(cufftComplex),
            cudaMemcpyDeviceToDevice, stream
        ));

        unscale<<<grid, block, 0, stream>>>(d_img, width, height, width);
    }

    checkError(cudaMemcpyAsync(
        h_img, d_img, height * width * sizeof(float),
        cudaMemcpyDeviceToHost, stream
    ));
}


void IlsInstance::compute(
    const float * h_input, int istride,
    float * h_output, int ostride
) noexcept {

    vs_bitblt(
        h_img, width * sizeof(float),
        h_input, istride * sizeof(float),
        width * sizeof(float), height
    );

    if (graphexec) {
        checkError(cudaGraphLaunch(graphexec, stream));
    } else {
        launch();
    }
    checkError(cudaStreamSynchronize(stream));

    vs_bitblt(
        h_output, ostride * sizeof(float),
        h_img, width * sizeof(float),
        width * sizeof(float), height
    );
}


void IlsInstance::release() noexcept {
    cudaFreeHost(h_img);
    cudaFree(d_img);
    cudaFree(d_tmp);
    cudaFree(d_normin1);
    cudaFree(d_normin1_backup);
    cudaFree(d_normin2);
    if (graphexec) {
        cudaGraphExecDestroy(graphexec);
    }
    cudaStreamDestroy(stream);
    cufftDestroy(irfft2d_plan);
    cufftDestroy(rfft2d_plan);
}
