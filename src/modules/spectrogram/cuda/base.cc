#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Spectrogram<D, T>::Impl {
    std::vector<U64> decayGrid;
    std::vector<U64> riseGrid;

    std::vector<U64> decayBlock;
    std::vector<U64> riseBlock;

    std::vector<void*> decayArguments;
    std::vector<void*> riseArguments;

    Tensor<Device::CUDA, T> input;

    // Pointer storage for kernel arguments
    void* inputPtr = nullptr;
    void* frequencyBinsPtr = nullptr;
};

template<Device D, typename T>
Spectrogram<D, T>::Spectrogram() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Spectrogram<D, T>::~Spectrogram() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Spectrogram<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Spectrogram compute core using CUDA backend.");

    // Initialize kernel input fallback tensor (will be allocated on first use if needed)

    // Create CUDA kernel.

    ctx.cuda->createKernel("decay", R"""(
        __global__ void decay(float* bins, float decayFactor, size_t size) {
            const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < size) {
                bins[id] *= decayFactor;
            }
        }
    )""");

    ctx.cuda->createKernel("rise",
                           R"""(
        #include "jetstream_window.cuh"
        __global__ void rise(const float* input, float* bins, size_t numberOfElements, size_t numberOfBatches, size_t height) {
            const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

            if (id >= numberOfElements) {
                return;
            }

            const float windowWeight = jst_window_hann(id, numberOfElements);

            for (size_t b = 0; b < numberOfBatches * numberOfElements; b += numberOfElements) {
                const size_t index = input[id + b] * height;

                if (index < height && index > 0) {
                    atomicAdd(&bins[id + (index * numberOfElements)], 0.02f * windowWeight);
                }
            }
        }
    )""",
                           {CUDA::KernelHeader::WINDOW});

    // Initialize kernel size.

    {
        U64 threadsPerBlock = 512;
        U64 blocksPerGrid = (gimpl->totalFrequencyBins + threadsPerBlock - 1) / threadsPerBlock;

        pimpl->decayGrid = { blocksPerGrid, 1, 1 };
        pimpl->decayBlock = { threadsPerBlock, 1, 1 };
    }

    {
        U64 threadsPerBlock = 512;
        U64 blocksPerGrid = (gimpl->numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

        pimpl->riseGrid = { blocksPerGrid, 1, 1 };
        pimpl->riseBlock = { threadsPerBlock, 1, 1 };
    }

    auto refreshFrequencyBinsPointer = [&]() {
        const auto base = reinterpret_cast<const uint8_t*>(gimpl->frequencyBins.data());
        pimpl->frequencyBinsPtr = const_cast<uint8_t*>(base) + gimpl->frequencyBins.offset_bytes();
    };
    refreshFrequencyBinsPointer();

    // Initialize kernel arguments.

    pimpl->decayArguments = {
        &pimpl->frequencyBinsPtr,
        &gimpl->decayFactor,
        &gimpl->totalFrequencyBins,
    };

    pimpl->riseArguments = {
        &pimpl->inputPtr,  // Address of pointer (will be updated in compute())
        &pimpl->frequencyBinsPtr,
        &gimpl->numberOfElements,
        &gimpl->numberOfBatches,
        &config.height,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::compute(const Context& ctx) {
    // Determine which tensor to use for CUDA kernel
    // D is Device::CUDA in this backend implementation

    auto tensorDataPointer = [](const auto& tensor) -> void* {
        const auto base = reinterpret_cast<const uint8_t*>(tensor.data());
        return const_cast<uint8_t*>(base) + tensor.offset_bytes();
    };

    // Update frequency bins pointer (in case tensor was reallocated)
    pimpl->frequencyBinsPtr = tensorDataPointer(gimpl->frequencyBins);

    if (input.buffer.contiguous()) {
        // Input is already contiguous, use directly
        pimpl->inputPtr = tensorDataPointer(input.buffer);
    } else {
        // Need to copy to CUDA fallback tensor (for non-contiguous data)
        if (pimpl->input.size() == 0 || pimpl->input.shape() != input.buffer.shape()) {
            pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
        }
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
        pimpl->inputPtr = tensorDataPointer(pimpl->input);
    }

    JST_CHECK(ctx.cuda->launchKernel("decay",
                                     pimpl->decayGrid,
                                     pimpl->decayBlock,
                                     pimpl->decayArguments.data()));

    JST_CHECK(ctx.cuda->launchKernel("rise",
                                     pimpl->riseGrid,
                                     pimpl->riseBlock,
                                     pimpl->riseArguments.data()));

    return Result::SUCCESS;
}

JST_SPECTROGRAM_CUDA(JST_INSTANTIATION)
JST_SPECTROGRAM_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
