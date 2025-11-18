#include "jetstream/compute/graph/cuda.hh"
#include "jetstream/backend/devices/cuda/helpers.hh"

#include <nvrtc.h>

#include <unordered_set>

namespace Jetstream {

namespace {

struct KernelHeaderDefinition {
    CUDA::KernelHeader type;
    const char* name;
    const char* source;
};

constexpr char kComplexHeaderName[] = "jetstream_complex.cuh";
constexpr char kComplexHeaderSource[] = R"(
#ifndef JETSTREAM_COMPLEX_HEADER
#define JETSTREAM_COMPLEX_HEADER

__device__ inline float2 jst_complex_make(float real, float imag) {
    return float2{real, imag};
}

__device__ inline float2 jst_complex_add(const float2& lhs, const float2& rhs) {
    return float2{lhs.x + rhs.x, lhs.y + rhs.y};
}

__device__ inline float2 jst_complex_mul(const float2& lhs, const float2& rhs) {
    return float2{
        (lhs.x * rhs.x) - (lhs.y * rhs.y),
        (lhs.x * rhs.y) + (lhs.y * rhs.x)
    };
}

__device__ inline float2 jst_complex_scale(const float2& value, float scalar) {
    return float2{value.x * scalar, value.y * scalar};
}

__device__ inline float jst_complex_abs(const float2& value) {
    return sqrtf((value.x * value.x) + (value.y * value.y));
}

__device__ inline float2 jst_complex_conj(const float2& value) {
    return float2{value.x, -value.y};
}

#endif  // JETSTREAM_COMPLEX_HEADER
)";

constexpr KernelHeaderDefinition kComplexHeaderDefinition{
    CUDA::KernelHeader::COMPLEX,
    kComplexHeaderName,
    kComplexHeaderSource,
};

constexpr char kTensorHeaderName[] = "jetstream_tensor.cuh";
constexpr char kTensorHeaderSource[] = R"(
#ifndef JETSTREAM_TENSOR_HEADER
#define JETSTREAM_TENSOR_HEADER

#ifndef JST_MAX_TENSOR_RANK
#define JST_MAX_TENSOR_RANK 8
#endif

__device__ inline void jst_tensor_index(size_t id,
                                        size_t rank,
                                        const size_t* shape,
                                        size_t* coords) {
    for (size_t i = 0; i < rank; ++i) {
        coords[i] = 0;
    }
    for (int idx = static_cast<int>(rank) - 1; idx >= 0; --idx) {
        coords[idx] = id % shape[idx];
        id /= shape[idx];
    }
}

__device__ inline size_t jst_tensor_offset(const size_t* coords,
                                           const size_t* strides,
                                           size_t rank) {
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

#endif  // JETSTREAM_TENSOR_HEADER
)";

constexpr KernelHeaderDefinition kTensorHeaderDefinition{
    CUDA::KernelHeader::TENSOR,
    kTensorHeaderName,
    kTensorHeaderSource,
};

constexpr char kWindowHeaderName[] = "jetstream_window.cuh";
constexpr char kWindowHeaderSource[] = R"(
#ifndef JETSTREAM_WINDOW_HEADER
#define JETSTREAM_WINDOW_HEADER

__device__ inline float jst_window_hann(size_t i, size_t size) {
    if (size <= 1) {
        return 1.0f;
    }
    return 0.5f * (1.0f - cosf((2.0f * CUDART_PI_F * i) / static_cast<float>(size - 1)));
}

__device__ inline float jst_window_hamming(size_t i, size_t size) {
    if (size <= 1) {
        return 1.0f;
    }
    return 0.54f - 0.46f * cosf((2.0f * CUDART_PI_F * i) / static_cast<float>(size - 1));
}

__device__ inline float jst_window_blackman(size_t i, size_t size) {
    if (size <= 1) {
        return 1.0f;
    }
    const float a0 = 0.42f;
    const float a1 = 0.5f;
    const float a2 = 0.08f;
    const float phase = (2.0f * CUDART_PI_F * i) / static_cast<float>(size - 1);
    return a0 - a1 * cosf(phase) + a2 * cosf(2.0f * phase);
}

#endif  // JETSTREAM_WINDOW_HEADER
)";

constexpr KernelHeaderDefinition kWindowHeaderDefinition{
    CUDA::KernelHeader::WINDOW,
    kWindowHeaderName,
    kWindowHeaderSource,
};

const KernelHeaderDefinition* FindHeaderDefinition(CUDA::KernelHeader header) {
    switch (header) {
        case CUDA::KernelHeader::COMPLEX:
            return &kComplexHeaderDefinition;
        case CUDA::KernelHeader::TENSOR:
            return &kTensorHeaderDefinition;
        case CUDA::KernelHeader::WINDOW:
            return &kWindowHeaderDefinition;
        default:
            return nullptr;
    }
}

}  // namespace

struct CUDA::Impl {
    struct Kernel {
        CUfunction function;
        CUmodule module;
    };

    U64 block_in_context;

    std::unordered_map<U64, std::unordered_map<std::string, Kernel>> kernels;
};

CUDA::CUDA() {
    context = std::make_shared<Compute::Context>();
    context->cuda = this;

    pimpl = std::make_unique<Impl>();
}

CUDA::~CUDA() {
    context.reset();
    pimpl.reset();
}

Result CUDA::create() {
    JST_DEBUG("Creating new CUDA compute graph.");

    // Create CUDA stream.

    JST_CUDA_CHECK(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking), [&]{
        JST_ERROR("[CUDA] Can't create stream: {}", err);
    });

    // Create blocks.

    for (U64 i = 0; i < computeUnits.size(); i++) {
        pimpl->block_in_context = i;

        JST_CHECK(computeUnits[i].block->createCompute(*context));
    }

    return Result::SUCCESS;
}

Result CUDA::computeReady() {
    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->computeReady());
    }
    return Result::SUCCESS;
}

Result CUDA::compute(std::unordered_set<U64>& yielded) {
    // Execute blocks.

    for (U64 i = 0; i < computeUnits.size(); i++) {
        auto& computeUnit = computeUnits[i];

        if (Graph::ShouldYield(yielded, computeUnit.inputSet)) {
            Graph::YieldCompute(yielded, computeUnit.outputSet);
            continue;
        }

        pimpl->block_in_context = i;

        const auto& res = computeUnit.block->compute(*context);

        if (res == Result::SUCCESS) {
            JST_CUDA_CHECK(cudaGetLastError(), [&]{
                JST_ERROR("[CUDA] Module kernel execution failed: {}", err);
            });

            continue;
        }

        if (res == Result::YIELD) {
            Graph::YieldCompute(yielded, computeUnit.outputSet);
            continue;
        }

        JST_CHECK(res);
    }

    // Wait for all blocks to finish.

    JST_CUDA_CHECK(cudaStreamSynchronize(_stream), [&]{
        JST_ERROR("[CUDA] Can't synchronize stream: {}", err);
    });

    return Result::SUCCESS;
}

Result CUDA::destroy() {
    // Destroy blocks.

    for (const auto& computeUnit : computeUnits) {
        JST_CHECK(computeUnit.block->destroyCompute(*context));
    }
    computeUnits.clear();

    // Destroy kernels.

    for (U64 i = 0; i < computeUnits.size(); i++) {
        pimpl->block_in_context = i;

        std::vector<std::string> kernel_names;
        for (const auto& [name, _] : pimpl->kernels[i]) {
            kernel_names.push_back(name);
        }
        for (const auto& name : kernel_names) {
            JST_CHECK(destroyKernel(name));
        }
    }

    // Destroy CUDA stream.

    JST_CUDA_CHECK(cudaStreamDestroy(_stream), [&]{
        JST_ERROR("[CUDA] Can't destroy stream: {}", err);
    });

    return Result::SUCCESS;
}

Result CUDA::createKernel(const std::string& name, 
                          const std::string& source,
                          const std::vector<KernelHeader>& headers) {
    if (pimpl->kernels[pimpl->block_in_context].contains(name)) {
        JST_ERROR("[CUDA] Kernel with name '{}' already exists.", name);
    }
    auto& kernel = pimpl->kernels[pimpl->block_in_context][name];

    // Create program.

    std::vector<const char*> headerSourcePtrs;
    std::vector<const char*> headerNamePtrs;
    headerSourcePtrs.reserve(headers.size());
    headerNamePtrs.reserve(headers.size());

    std::unordered_set<KernelHeader> uniqueHeaders;
    for (const auto& header : headers) {
        if (header == KernelHeader::NONE) {
            continue;
        }
        if (!uniqueHeaders.insert(header).second) {
            continue;
        }
        const auto* definition = FindHeaderDefinition(header);
        if (!definition) {
            JST_WARN("[CUDA] Unknown kernel header requested: {}", static_cast<int>(header));
            continue;
        }
        headerSourcePtrs.push_back(definition->source);
        headerNamePtrs.push_back(definition->name);
    }

    nvrtcProgram program;
    JST_NVRTC_CHECK(nvrtcCreateProgram(&program,
                                       source.c_str(),
                                       nullptr,
                                       headerSourcePtrs.size(),
                                       headerSourcePtrs.empty() ? nullptr : headerSourcePtrs.data(),
                                       headerNamePtrs.empty() ? nullptr : headerNamePtrs.data()), [&]{
        JST_ERROR("[CUDA] Can't create program: {}", err);
    });

    // Add name expression.

    JST_NVRTC_CHECK(nvrtcAddNameExpression(program, name.c_str()), [&]{
        JST_ERROR("[CUDA] Can't add name expression: {}", err);
    });

    // Compile program.

    const std::vector<const char*> options = {
        "--gpu-architecture=compute_86",
        "--std=c++14"
    };

    JST_NVRTC_CHECK(nvrtcCompileProgram(program, options.size(), options.data()), [&]{
        size_t logSize;
        nvrtcGetProgramLogSize(program, &logSize);
        std::vector<char> log(logSize);
        nvrtcGetProgramLog(program, log.data());
        JST_ERROR("[CUDA] Can't compile program:\n{}", log.data());
    });

    // Get PTX.

    size_t ptxSize;
    JST_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptxSize), [&]{
        JST_ERROR("[CUDA] Can't get PTX size: {}", err);
    });

    std::vector<char> ptx(ptxSize);
    JST_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()), [&]{
        JST_ERROR("[CUDA] Can't get PTX: {}", err);
    });

    // Print PTX.

    JST_TRACE("[CUDA] Generated kernel PTX ({}):\n{}", name, ptx.data());

    // Get lowered name.

    const char* loweredName;
    JST_NVRTC_CHECK(nvrtcGetLoweredName(program, name.c_str(), &loweredName), [&]{
        JST_ERROR("[CUDA] Can't get lowered name: {}", err);
    });
    
    // Create module.

    JST_CUDA_CHECK(cuModuleLoadData(&kernel.module, ptx.data()), [&]{
        JST_ERROR("[CUDA] Can't load module: {}", err);
    });

    // Get function.

    JST_CUDA_CHECK(cuModuleGetFunction(&kernel.function, kernel.module, loweredName), [&]{
        JST_ERROR("[CUDA] Can't get function: {}", err);
    });

    // Destroy program.

    JST_NVRTC_CHECK(nvrtcDestroyProgram(&program), [&]{
        JST_ERROR("[CUDA] Can't destroy program: {}", err);
    });

    return Result::SUCCESS;
}

Result CUDA::destroyKernel(const std::string& name) {
    auto& kernel = pimpl->kernels[pimpl->block_in_context][name];

    JST_CUDA_CHECK(cuModuleUnload(kernel.module), [&]{
        JST_ERROR("[CUDA] Can't unload module: {}", err);
    });

    pimpl->kernels[pimpl->block_in_context].erase(name);

    return Result::SUCCESS;
}

Result CUDA::launchKernel(const std::string& name, 
                          const std::vector<U64>& grid,
                          const std::vector<U64>& block,
                          void** arguments) {
    const auto& kernel = pimpl->kernels.at(pimpl->block_in_context).at(name);

    JST_CUDA_CHECK(cuLaunchKernel(kernel.function, 
                                  grid[0], grid[1], grid[2], 
                                  block[0], block[1], block[2],
                                  0, _stream, arguments, 0), [&]{
        JST_ERROR("[CUDA] Can't launch kernel '{}': {}", name, err);
    });

    return Result::SUCCESS;
}

}  // namespace Jetstream
