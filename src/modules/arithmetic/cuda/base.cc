#include <regex>

#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Arithmetic<D, T>::Impl {
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

    Tensor<Device::CUDA, T> input;

    Tensor<D, T> broadcasted_output;

    int operation;
};

template<Device D, typename T>
Arithmetic<D, T>::Arithmetic() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Arithmetic<D, T>::~Arithmetic() {
    pimpl.reset();
}

template<Device D, typename T>
Result Arithmetic<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Arithmetic compute core using CUDA backend.");

    // Create CUDA kernel.

    std::string kernel = R"""(
        #include "jetstream_tensor.cuh"
        struct Meta {
            void* ptr;
            size_t rank;
            size_t shape[8];
            size_t strides[8];
        };

        enum ArithmeticOperation {
            kAdd = 0,
            kSub = 1,
            kMul = 2,
            kDiv = 3,
        };

        __device__ inline void atomic_mul(float* address, float value) {
            float old = *address;
            float assumed;
            do {
                assumed = old;
                const float updated = assumed * value;
                old = __uint_as_float(atomicCAS(reinterpret_cast<unsigned int*>(address),
                                                __float_as_uint(assumed),
                                                __float_as_uint(updated)));
            } while (__float_as_uint(assumed) != __float_as_uint(old));
        }

        __device__ inline void atomic_div(float* address, float value) {
            float old = *address;
            float assumed;
            do {
                assumed = old;
                const float updated = assumed / value;
                old = __uint_as_float(atomicCAS(reinterpret_cast<unsigned int*>(address),
                                                __float_as_uint(assumed),
                                                __float_as_uint(updated)));
            } while (__float_as_uint(assumed) != __float_as_uint(old));
        }

        __device__ inline float2 complex_mul(const float2& lhs, const float2& rhs) {
            return float2{(lhs.x * rhs.x) - (lhs.y * rhs.y),
                          (lhs.x * rhs.y) + (lhs.y * rhs.x)};
        }

        __device__ inline float2 complex_div(const float2& lhs, const float2& rhs) {
            const float denom = (rhs.x * rhs.x) + (rhs.y * rhs.y) + 1e-20f;
            return float2{((lhs.x * rhs.x) + (lhs.y * rhs.y)) / denom,
                          ((lhs.y * rhs.x) - (lhs.x * rhs.y)) / denom};
        }

        union Float2Uint64 {
            unsigned long long int bits;
            float2 value;
        };

        __device__ inline void atomic_mul(float2* address, const float2& value) {
            auto* raw = reinterpret_cast<unsigned long long int*>(address);
            unsigned long long int old = *raw;
            while (true) {
                Float2Uint64 expected;
                expected.bits = old;
                const float2 updated = complex_mul(expected.value, value);
                Float2Uint64 desired;
                desired.value = updated;
                old = atomicCAS(raw, expected.bits, desired.bits);
                if (expected.bits == old) {
                    break;
                }
            }
        }

        __device__ inline void atomic_div(float2* address, const float2& value) {
            auto* raw = reinterpret_cast<unsigned long long int*>(address);
            unsigned long long int old = *raw;
            while (true) {
                Float2Uint64 expected;
                expected.bits = old;
                const float2 updated = complex_div(expected.value, value);
                Float2Uint64 desired;
                desired.value = updated;
                old = atomicCAS(raw, expected.bits, desired.bits);
                if (expected.bits == old) {
                    break;
                }
            }
        }

        __device__ inline void apply_operation(float* target, float value, int operation) {
            switch (operation) {
                case kAdd:
                    atomicAdd(target, value);
                    break;
                case kSub:
                    atomicAdd(target, -value);
                    break;
                case kMul:
                    atomic_mul(target, value);
                    break;
                case kDiv:
                    atomic_div(target, value);
                    break;
            }
        }

        __device__ inline void apply_operation(float2* target, const float2& value, int operation) {
            switch (operation) {
                case kAdd:
                    atomicAdd(&target->x, value.x);
                    atomicAdd(&target->y, value.y);
                    break;
                case kSub:
                    atomicAdd(&target->x, -value.x);
                    atomicAdd(&target->y, -value.y);
                    break;
                case kMul:
                    atomic_mul(target, value);
                    break;
                case kDiv:
                    atomic_div(target, value);
                    break;
            }
        }

        __global__ void arithmetic(Meta input, Meta output, size_t size, int operation) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;

            // Return if ID is out of bounds.

            if (id >= size) {
                return;
            }

            // Calculate input and output offsets using shared tensor helpers.

            size_t coords[JST_MAX_TENSOR_RANK];
            jst_tensor_index(id, input.rank, input.shape, coords);
            size_t input_offset = jst_tensor_offset(coords, input.strides, input.rank);
            size_t output_offset = jst_tensor_offset(coords, output.strides, output.rank);

            // Reinterpret input and output pointers.

            [//CAST//]

            // Perform arithmetic operation.

            [//OP//]
        }
    )""";

    if constexpr (std::is_same_v<T, F32>) {
        const std::string cast = R"""(
            const auto* input_ptr = reinterpret_cast<float*>(input.ptr);
            auto* output_ptr = reinterpret_cast<float*>(output.ptr);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/CAST\/\/\])"), cast);

        const std::string operation = R"""(
            apply_operation(&output_ptr[output_offset], input_ptr[input_offset], operation);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    } else if constexpr (std::is_same_v<T, CF32>) {
        const std::string cast = R"""(
            const auto* input_ptr = reinterpret_cast<float2*>(input.ptr);
            auto* output_ptr = reinterpret_cast<float2*>(output.ptr);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/CAST\/\/\])"), cast);

        const std::string operation = R"""(
            apply_operation(&output_ptr[output_offset], input_ptr[input_offset], operation);
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    }

    ctx.cuda->createKernel("arithmetic", kernel, {CUDA::KernelHeader::TENSOR});

    // Initialize kernel size.

    U64 threadsPerBlock = 512;
    U64 blocksPerGrid = (input.buffer.size() + threadsPerBlock - 1) / threadsPerBlock;

    pimpl->grid = { blocksPerGrid, 1, 1 };
    pimpl->block = { threadsPerBlock, 1, 1 };

    // Initialize kernel input.

    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        pimpl->input = Tensor<Device::CUDA, T>(input.buffer.shape());
    } else {
        pimpl->input = input.buffer;
    }

    // Initialize kernel arguments.

    pimpl->inputMeta = {
        reinterpret_cast<uint8_t*>(pimpl->input.data()) + pimpl->input.offset_bytes(),
        pimpl->input.rank(),
        {},
        {},
    };

    for (U64 i = 0; i < pimpl->input.rank(); i++) {
        pimpl->inputMeta.shape[i] = pimpl->input.shape()[i];
        pimpl->inputMeta.strides[i] = pimpl->input.stride()[i];
    }

    pimpl->outputMeta = {
        reinterpret_cast<uint8_t*>(pimpl->broadcasted_output.data()) + pimpl->broadcasted_output.offset_bytes(),
        pimpl->broadcasted_output.rank(),
        {},
        {},
    };

    for (U64 i = 0; i < pimpl->broadcasted_output.rank(); i++) {
        pimpl->outputMeta.shape[i] = pimpl->broadcasted_output.shape()[i];
        pimpl->outputMeta.strides[i] = pimpl->broadcasted_output.stride()[i];
    }

    pimpl->size = input.buffer.size();

    pimpl->operation = static_cast<int>(config.operation);

    pimpl->arguments = {
        &pimpl->inputMeta,
        &pimpl->outputMeta,
        &pimpl->size,
        &pimpl->operation,
    };

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Arithmetic<D, T>::compute(const Context& ctx) {
    if (!input.buffer.device_native() && input.buffer.contiguous()) {
        JST_CHECK(Memory::Copy(pimpl->input, input.buffer, ctx.cuda->stream()));
    }

    // Update input pointer in case tensor was reallocated
    pimpl->inputMeta.ptr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(pimpl->input.data())) + pimpl->input.offset_bytes();

    pimpl->operation = static_cast<int>(config.operation);

    JST_CUDA_CHECK(cudaMemsetAsync(output.buffer.data(), 0, output.buffer.size_bytes(), ctx.cuda->stream()), [&]{
        JST_ERROR("Failed to clear output buffer: {}", err);
    });

    JST_CHECK(ctx.cuda->launchKernel("arithmetic",
                                     pimpl->grid,
                                     pimpl->block,
                                     pimpl->arguments.data()));

    return Result::SUCCESS;
}

JST_ARITHMETIC_CUDA(JST_INSTANTIATION)
JST_ARITHMETIC_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
