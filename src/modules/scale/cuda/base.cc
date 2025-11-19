#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Scale<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    std::vector<void*> arguments;

    struct Meta {
        void* ptr;
        size_t rank;
        size_t shape[8];
        size_t strides[8];
    };

    Tensor<Device::CUDA, T> input;
    Meta inputMeta;
    Meta outputMeta;

    F32 scalingCoeff;
    F32 offsetCoeff;
    U64 numberOfElements;
};

template<Device D, typename T>
Scale<D, T>::Scale() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Scale<D, T>::~Scale() {
    impl.reset();
}

template<Device D, typename T>
Result Scale<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Scale compute core using CUDA backend.");

    // Create CUDA kernel.

    ctx.cuda->createKernel("scale",
                           R"""(
        #include "jetstream_tensor.cuh"
        struct Meta {
            void* ptr;
            size_t rank;
            size_t shape[8];
            size_t strides[8];
        };

        __device__ inline size_t tensor_offset(const Meta& meta, size_t index) {
            size_t coords[JST_MAX_TENSOR_RANK];
            jst_tensor_index(index, meta.rank, meta.shape, coords);
            return jst_tensor_offset(coords, meta.strides, meta.rank);
        }

        __global__ void scale(const Meta input, const Meta output, float scalingCoeff, float offsetCoeff, size_t size) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id < size) {
                const size_t inputOffset = tensor_offset(input, id);
                const size_t outputOffset = tensor_offset(output, id);
                const float* inputPtr = reinterpret_cast<const float*>(input.ptr);
                float* outputPtr = reinterpret_cast<float*>(output.ptr);
                outputPtr[outputOffset] = inputPtr[inputOffset] * scalingCoeff + offsetCoeff;
            }
        }
    )""",
                           {CUDA::KernelHeader::TENSOR});

    // Initialize kernel size.

    U64 threadsPerBlock = 512;
    U64 blocksPerGrid = (impl->numberOfElements + threadsPerBlock - 1) / threadsPerBlock;

    impl->grid = { blocksPerGrid, 1, 1 };
    impl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        impl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        impl->input = input.buffer;
    }

    {
        auto& tensor = impl->input;
        impl->inputMeta = {
            reinterpret_cast<uint8_t*>(tensor.data()) + tensor.offset_bytes(),
            tensor.rank(),
            {},
            {},
        };

        for (U64 i = 0; i < tensor.rank(); ++i) {
            impl->inputMeta.shape[i] = tensor.shape()[i];
            impl->inputMeta.strides[i] = tensor.stride()[i];
        }
    }

    impl->outputMeta = {
        reinterpret_cast<uint8_t*>(output.buffer.data()) + output.buffer.offset_bytes(),
        output.buffer.rank(),
        {},
        {},
    };
    for (U64 i = 0; i < output.buffer.rank(); ++i) {
        impl->outputMeta.shape[i] = output.buffer.shape()[i];
        impl->outputMeta.strides[i] = output.buffer.stride()[i];
    }

    // Initialize kernel arguments.

    impl->arguments = {
        &impl->inputMeta,
        &impl->outputMeta,
        &impl->scalingCoeff,
        &impl->offsetCoeff,
        &impl->numberOfElements,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Scale<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(Memory::Copy(impl->input, input.buffer, ctx.cuda->stream()));
    }

    // Update input pointer in case tensor was reallocated
    auto& tensor = impl->input;
    impl->inputMeta.ptr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(tensor.data())) + tensor.offset_bytes();

    JST_CHECK(ctx.cuda->launchKernel("scale",
                                     impl->grid,
                                     impl->block,
                                     impl->arguments.data()));

    return Result::SUCCESS;
}

JST_SCALE_CUDA(JST_INSTANTIATION)
JST_SCALE_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
