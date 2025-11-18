#include <algorithm>

#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

namespace {
constexpr U64 kLineplotThreadsPerBlock = 256;
}

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> argumentsLineplot;

    struct Meta {
        void* ptr;
        size_t rank;
        size_t shape[8];
        size_t strides[8];
    };

    Meta inputMeta;
    Tensor<Device::CUDA, T> input;
};

template<Device D, typename T>
Lineplot<D, T>::Lineplot() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Lineplot<D, T>::~Lineplot() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Lineplot compute core using CUDA backend.");

    // Create CUDA kernel.

    ctx.cuda->createKernel("lineplot",
                           R"""(
        #include "jetstream_tensor.cuh"
        struct Meta {
            void* ptr;
            size_t rank;
            size_t shape[8];
            size_t strides[8];
        };

        __device__ inline float tensor_read(const Meta& meta, size_t index) {
            size_t coords[JST_MAX_TENSOR_RANK];
            jst_tensor_index(index, meta.rank, meta.shape, coords);
            const auto* ptr = reinterpret_cast<const float*>(meta.ptr);
            return ptr[jst_tensor_offset(coords, meta.strides, meta.rank)];
        }

        constexpr int kBlockWidth = 256;

        __global__ void lineplot(const Meta input, float2* output, float normalizationFactor, size_t numberOfBatches, size_t numberOfElements, size_t averaging, size_t decimation) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ float sharedAverage[kBlockWidth];
            __shared__ float sharedX[kBlockWidth];

            if (id < numberOfElements) {
                sharedAverage[threadIdx.x] = output[id].y;
            }
            __syncthreads();

            if (id >= numberOfElements) {
                return;
            }

            // Compute average amplitude within a batch.
            float amplitude = 0.0f;
            const size_t baseIndex = id * decimation;
            for (size_t i = 0; i < numberOfBatches; ++i) {
                const size_t sampleIndex = baseIndex + (i * numberOfElements);
                amplitude += tensor_read(input, sampleIndex);
            }
            amplitude = (amplitude * normalizationFactor) - 1.0f;

            // Calculate moving average using shared memory staging.
            float average = sharedAverage[threadIdx.x];
            average -= average / averaging;
            average += amplitude / averaging;

            sharedX[threadIdx.x] = id * 2.0f / (numberOfElements - 1) - 1.0f;
            sharedAverage[threadIdx.x] = average;

            __syncthreads();

            output[id].x = sharedX[threadIdx.x];
            output[id].y = sharedAverage[threadIdx.x];
        }
    )""",
                           {CUDA::KernelHeader::TENSOR});

    // Initialize kernel size.

    const U64 threadsPerBlock = std::min<U64>(kLineplotThreadsPerBlock,
                                              gimpl->numberOfElements == 0 ? 1 : gimpl->numberOfElements);
    U64 blocksPerGrid = (gimpl->numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    {
        auto& tensor = pimpl->input;
        pimpl->inputMeta = {
            reinterpret_cast<uint8_t*>(tensor.data()) + tensor.offset_bytes(),
            tensor.rank(),
            {},
            {},
        };

        for (U64 i = 0; i < tensor.rank(); ++i) {
            pimpl->inputMeta.shape[i] = tensor.shape()[i];
            pimpl->inputMeta.strides[i] = tensor.stride()[i];
        }
    }

    // Initialize kernel arguments.

    pimpl->argumentsLineplot = {
        &pimpl->inputMeta,
        gimpl->signalPoints.data_ptr(),
        &gimpl->normalizationFactor,
        &gimpl->numberOfBatches,
        &gimpl->numberOfElements,
        &config.averaging,
        &config.decimation,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    const U64 threadsPerBlock = std::min<U64>(kLineplotThreadsPerBlock,
                                              gimpl->numberOfElements == 0 ? 1 : gimpl->numberOfElements);
    pimpl->block[0] = threadsPerBlock;
    pimpl->grid[0] = (gimpl->numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    JST_CHECK(ctx.cuda->launchKernel("lineplot",
                                     pimpl->grid,
                                     pimpl->block,
                                     pimpl->argumentsLineplot.data()));

    gimpl->updateSignalPointsFlag = true;

    return Result::SUCCESS;
}

JST_LINEPLOT_CUDA(JST_INSTANTIATION)
JST_LINEPLOT_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
