#include <algorithm>
#include <regex>

#include "../generic.cc"

#include "jetstream/memory/devices/cuda/copy.hh"

namespace Jetstream {

template<Device D, typename T>
struct Multiply<D, T>::Impl {
    struct Meta {
        void* ptr;
        size_t rank;
        size_t shape[8];
        size_t strides[8];
    };

    std::vector<U64> grid;
    std::vector<U64> block;

    Meta factorAMeta {};
    Meta factorBMeta {};
    Meta productMeta {};
    U64 elementCount = 0;

    Tensor<Device::CUDA, T> factorAFallback;
    Tensor<Device::CUDA, T> factorBFallback;
    Tensor<Device::CUDA, T> productFallback;
};

template<Device D, typename T>
Multiply<D, T>::Multiply() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Multiply<D, T>::~Multiply() {
    impl.reset();
}

template<Device D, typename T>
Result Multiply<D, T>::createCompute(const Context& ctx) {
    JST_TRACE("Create Multiply compute core using CUDA backend.");

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

        __global__ void multiply(Meta factorA, Meta factorB, Meta product, size_t size) {
            size_t id = blockIdx.x * blockDim.x + threadIdx.x;
            if (id >= size) {
                return;
            }

            const size_t aIdx = tensor_offset(factorA, id);
            const size_t bIdx = tensor_offset(factorB, id);
            const size_t oIdx = tensor_offset(product, id);

            [//OP//]
        }
    )""";

    if constexpr (std::is_same_v<T, F32>) {
        const std::string operation = R"""(
            const auto* aPtr = reinterpret_cast<const float*>(factorA.ptr);
            const auto* bPtr = reinterpret_cast<const float*>(factorB.ptr);
            auto* oPtr = reinterpret_cast<float*>(product.ptr);
            oPtr[oIdx] = aPtr[aIdx] * bPtr[bIdx];
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    } else if constexpr (std::is_same_v<T, CF32>) {
        const std::string operation = R"""(
            const auto* aPtr = reinterpret_cast<const float2*>(factorA.ptr);
            const auto* bPtr = reinterpret_cast<const float2*>(factorB.ptr);
            auto* oPtr = reinterpret_cast<float2*>(product.ptr);
            const float2 lhs = aPtr[aIdx];
            const float2 rhs = bPtr[bIdx];
            oPtr[oIdx] = float2{
                (lhs.x * rhs.x) - (lhs.y * rhs.y),
                (lhs.x * rhs.y) + (lhs.y * rhs.x)
            };
        )""";
        kernel = std::regex_replace(kernel, std::regex(R"(\[\/\/OP\/\/\])"), operation);
    }

    ctx.cuda->createKernel("multiply", kernel, {CUDA::KernelHeader::TENSOR});

    impl->grid = {1, 1, 1};
    impl->block = {512, 1, 1};

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Multiply<D, T>::compute(const Context& ctx) {
    auto prepareTensor = [&](const Tensor<D, T>& source,
                             Tensor<Device::CUDA, T>& fallback,
                             typename Impl::Meta& meta) -> const Tensor<Device::CUDA, T>& {
        const Tensor<Device::CUDA, T>* tensor = nullptr;
        if (source.device_native() && source.contiguous()) {
            tensor = &source;
        } else {
            if (fallback.size() == 0 || fallback.shape() != source.shape()) {
                fallback = Tensor<Device::CUDA, T>(source.shape());
            }
            JST_CHECK(Memory::Copy(fallback, source, ctx.cuda->stream()));
            tensor = &fallback;
        }

        meta.ptr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(tensor->data())) + tensor->offset_bytes();
        meta.rank = tensor->rank();
        for (size_t i = 0; i < tensor->rank(); ++i) {
            meta.shape[i] = tensor->shape()[i];
            meta.strides[i] = tensor->stride()[i];
        }

        return *tensor;
    };

    const auto& factorA = prepareTensor(input.factorA, impl->factorAFallback, impl->factorAMeta);
    const auto& factorB = prepareTensor(input.factorB, impl->factorBFallback, impl->factorBMeta);

    bool copyBack = false;
    const Tensor<Device::CUDA, T>* productTensor = nullptr;
    if (output.product.device_native() && output.product.contiguous()) {
        productTensor = reinterpret_cast<const Tensor<Device::CUDA, T>*>(&output.product);
    } else {
        if (impl->productFallback.size() == 0 || impl->productFallback.shape() != output.product.shape()) {
            impl->productFallback = Tensor<Device::CUDA, T>(output.product.shape());
        }
        productTensor = &impl->productFallback;
        copyBack = true;
    }

    impl->productMeta.ptr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(productTensor->data())) + productTensor->offset_bytes();
    impl->productMeta.rank = productTensor->rank();
    for (size_t i = 0; i < productTensor->rank(); ++i) {
        impl->productMeta.shape[i] = productTensor->shape()[i];
        impl->productMeta.strides[i] = productTensor->stride()[i];
    }

    impl->elementCount = productTensor->size();
    const U64 threadsPerBlock = impl->block[0];
    const U64 blocksPerGrid = (impl->elementCount + threadsPerBlock - 1) / threadsPerBlock;
    impl->grid[0] = std::max<U64>(1, blocksPerGrid);

    void* args[] = {
        &impl->factorAMeta,
        &impl->factorBMeta,
        &impl->productMeta,
        &impl->elementCount,
    };

    JST_CHECK(ctx.cuda->launchKernel("multiply",
                                     impl->grid,
                                     impl->block,
                                     args));

    if (copyBack) {
        JST_CHECK(Memory::Copy(output.product, impl->productFallback, ctx.cuda->stream()));
    }

    return Result::SUCCESS;
}

JST_MULTIPLY_CUDA(JST_INSTANTIATION)
JST_MULTIPLY_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
