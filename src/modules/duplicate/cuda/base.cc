#include <regex>

#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Duplicate<D, T>::Impl {
    std::vector<U64> grid;
    std::vector<U64> block;

    struct Meta {
        void* ptr;
        size_t rank;
        size_t shape[8];
        size_t strides[8];
    };

    Meta inputMeta;
    Meta outputMeta;
    U64 size;

    std::vector<void*> arguments;
};

template<Device D, typename T>
Duplicate<D, T>::Duplicate() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Duplicate<D, T>::~Duplicate() {
    pimpl.reset();
}

template<Device D, typename T>
Result Duplicate<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Duplicate compute core using CUDA backend.");

    // Create CUDA kernel.

    std::string kernel = R"""(
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

        __global__ void duplicate(Meta input, Meta output, size_t size) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;

            // Return if ID is out of bounds.

            if (id >= size) {
                return;
            }

            const size_t input_offset = tensor_offset(input, id);
            const size_t output_offset = tensor_offset(output, id);

            // Reinterpret input and output pointers.

            [//OP//]
        }
    )""";

    if constexpr (std::is_same_v<T, F32>) {
        const std::string operation = R"""(
            const auto* input_ptr = reinterpret_cast<float*>(input.ptr);
            auto* output_ptr = reinterpret_cast<float*>(output.ptr);

            output_ptr[output_offset] = input_ptr[input_offset];
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    } else if constexpr (std::is_same_v<T, CF32>) {
        const std::string operation = R"""(
            const auto* input_ptr = reinterpret_cast<float2*>(input.ptr);
            auto* output_ptr = reinterpret_cast<float2*>(output.ptr);

            output_ptr[output_offset] = input_ptr[input_offset];
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    }

    ctx.cuda->createKernel("duplicate", kernel, {CUDA::KernelHeader::TENSOR});

    // Initialize kernel size.

    U64 threadsPerBlock = 512;
    U64 blocksPerGrid = (input.buffer.size() + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel arguments.

    pimpl->inputMeta = {
        reinterpret_cast<uint8_t*>(input.buffer.data()) + input.buffer.offset_bytes(),
        input.buffer.rank(),
        {},
        {},
    };

    for (U64 i = 0; i < input.buffer.rank(); i++) {
        pimpl->inputMeta.shape[i] = input.buffer.shape()[i];
        pimpl->inputMeta.strides[i] = input.buffer.stride()[i];
    }

    pimpl->outputMeta = {
        reinterpret_cast<uint8_t*>(output.buffer.data()) + output.buffer.offset_bytes(),
        output.buffer.rank(),
        {},
        {},
    };

    for (U64 i = 0; i < output.buffer.rank(); i++) {
        pimpl->outputMeta.shape[i] = output.buffer.shape()[i];
        pimpl->outputMeta.strides[i] = output.buffer.stride()[i];
    }

    pimpl->size = input.buffer.size();

    pimpl->arguments = {
        &pimpl->inputMeta,
        &pimpl->outputMeta,
        &pimpl->size,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Duplicate<D, T>::compute(const Context& ctx) {
    if (input.buffer.contiguous()) {
        JST_CHECK(Memory::Copy(output.buffer, input.buffer, ctx.cuda->stream()));
    } else {
        JST_CHECK(ctx.cuda->launchKernel("duplicate", 
                                         pimpl->grid, 
                                         pimpl->block, 
                                         pimpl->arguments.data()));
    }

    return Result::SUCCESS;
}

JST_DUPLICATE_CUDA(JST_INSTANTIATION)
JST_DUPLICATE_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
