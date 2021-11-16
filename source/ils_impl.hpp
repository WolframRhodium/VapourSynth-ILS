#include <cufft.h>

struct IlsParams {
    int iteration;
    float c, p, eps, gamma, lambda;
    bool use_welsch;
};

struct IlsInstance {
    IlsParams param;
    int width, height;

    cufftHandle rfft2d_plan;
    cufftHandle irfft2d_plan;
    cudaStream_t stream;
    cudaGraphExec_t graphexec;
    cufftReal * d_img, * d_tmp;
    cufftComplex * d_normin1, * d_normin1_backup, * d_normin2;
    float * h_img;
    const float * d_denormin;

    void init(
        const IlsParams & param,
        int width, int height,
        bool use_cuda_graph,
        const float * d_denormin
    ) noexcept;

    void launch() noexcept;

    void compute(
        const float * h_input, int istride,
        float * h_output, int ostride
    ) noexcept;

    void release() noexcept;
};

float * init_denormin(
    int width, int height,
    float c, float lambda
) noexcept;
