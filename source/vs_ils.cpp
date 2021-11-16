#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <cuda_runtime_api.h>

#include "ils_impl.hpp"


struct TicketSemaphore {
    std::atomic<intptr_t> ticket {};
    std::atomic<intptr_t> current {};

    void acquire() noexcept {
        intptr_t tk { ticket.fetch_add(1, std::memory_order::acquire) };
        while (true) {
            intptr_t curr { current.load(std::memory_order::acquire) };
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};


struct IlsServer {
    int device_id;
    float * d_denormin;
    TicketSemaphore semaphore;

    std::mutex mutex;
    std::vector<IlsInstance> instances;
    std::vector<int> tickets;

    void init(const IlsParams & param,
        int width, int height,
        int device_id,
        int num_instances,
        bool use_cuda_graph
    ) noexcept {

        this->device_id = device_id;

        semaphore.current.store(num_instances - 1, std::memory_order::relaxed);

        cudaSetDevice(device_id);

        d_denormin = init_denormin(width, height, param.c, param.lambda);

        tickets.reserve(num_instances);
        instances.reserve(num_instances);
        for (int i = 0; i < num_instances; ++i) {
            tickets.emplace_back(i);
            IlsInstance instance;
            instance.init(param, width, height, use_cuda_graph, d_denormin);
            instances.push_back(std::move(instance));
        }
    }

    void compute(
        const float * h_input, int istride,
        float * h_output, int ostride
    ) noexcept {

        semaphore.acquire();

        int ticket = [&]() {
            std::lock_guard<std::mutex> lock { mutex };
            int ticket = tickets.back();
            tickets.pop_back();
            return ticket;
        }();

        auto & instance = instances[ticket];

        cudaSetDevice(device_id);
        instance.compute(h_input, istride, h_output, ostride);

        {
            std::lock_guard<std::mutex> lock { mutex };
            tickets.push_back(ticket);
        }

        semaphore.release();
    }

    ~IlsServer() noexcept {
        for (auto & instance : instances) {
            instance.release();
        }
    }
};


struct IlsData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    IlsServer server;
};


static void VS_CC ILSInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) {

    auto d = static_cast<IlsData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}


static const VSFrameRef *VS_CC ILSGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {

    auto d = static_cast<IlsData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);

        const int pl[] { 0, 1, 2 };
        const VSFrameRef * fr[] { nullptr, src, src };

        VSFrameRef * dst = vsapi->newVideoFrame2(
            d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core
        );

        d->server.compute(
            reinterpret_cast<const float *>(vsapi->getReadPtr(src, 0)),
            vsapi->getStride(src, 0) / sizeof(float),
            reinterpret_cast<float *>(vsapi->getWritePtr(dst, 0)),
            vsapi->getStride(dst, 0) / sizeof(float)
        );

        vsapi->freeFrame(src);

        return dst;
    }

    return nullptr;
}


static void VS_CC ILSFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<IlsData *>(instanceData);

    vsapi->freeNode(d->node);

    delete d;
}


static void VS_CC ILSCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) {
    auto d = std::make_unique<IlsData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);

    auto set_error = [&](const std::string & error_message) {
        vsapi->setError(out, ("ILS: " + error_message).c_str());
        vsapi->freeNode(d->node);
    };

    d->vi = vsapi->getVideoInfo(d->node);
    if (!isConstantFormat(d->vi) ||
        d->vi->format->sampleType != stFloat ||
        d->vi->format->bitsPerSample != 32
    ) {
        return set_error("only constant format 32bit float input supported");
    }

    auto load_param = [&]<typename T>(const char * name, T fallback) -> T {
        static_assert(std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, float>);

        int error;

        T val;
        if constexpr (std::is_same_v<T, float>) {
            val = static_cast<float>(vsapi->propGetFloat(in, name, 0, &error));
        } else if constexpr (std::is_same_v<T, int>) {
            val = int64ToIntS(vsapi->propGetInt(in, name, 0, &error));
        } else if constexpr (std::is_same_v<T, bool>) {
            val = !!vsapi->propGetInt(in, name, 0, &error);
        }

        if (error) {
            return fallback;
        }
        return val;
    };

    IlsParams param;
    param.lambda = load_param("lambda", 0.5f);
    param.iteration = load_param("iteration", 4);
    param.p = load_param("p", 0.8f);
    param.eps = load_param("eps", 0.0001f);
    param.gamma = load_param("gamma", 0.5f * param.p - 1.0f);
    param.c = load_param("c", param.p * powf(param.eps, param.gamma));
    param.use_welsch = load_param("use_welsch", false);

    int device_id = load_param("device_id", 0);
    int num_instances = load_param("num_streams", 2);
    bool use_cuda_graph = load_param("use_cuda_graph", true);

    d->server.init(
        param, d->vi->width, d->vi->height,
        device_id, num_instances, use_cuda_graph
    );

    vsapi->createFilter(
        in, out, "ILS",
        ILSInit, ILSGetFrame, ILSFree,
        fmParallel, 0, d.release(), core
    );
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) {

    configFunc(
        "com.wolframrhodium.ils", "ils",
        "CUDA implementation of Real-time Image Smoothing via Iterative Least Squares",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("ILS",
        "clip:clip;"
        "lambda:float:opt;"
        "iteration:int:opt;"
        "p:float:opt;"
        "eps:float:opt;"
        "gamma:float:opt;"
        "c:float:opt;"
        "use_welsch:int:opt;"
        "device_id:int:opt;"
        "num_streams:int:opt;"
        "use_cuda_graph:int:opt;",
        ILSCreate, nullptr, plugin
    );
}
