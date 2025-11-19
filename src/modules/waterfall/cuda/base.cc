#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct Waterfall<D, T>::Impl {};

template<Device D, typename T>
Waterfall<D, T>::Waterfall() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Waterfall<D, T>::~Waterfall() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Waterfall<D, T>::GImpl::underlyingCompute(Waterfall<D, T>& m, const Context& ctx) {
    const auto totalSize = m.input.buffer.size_bytes();
    const auto fftSize = numberOfElements * sizeof(T);
    const auto offset = inc * fftSize;
    const auto size = JST_MIN(totalSize, (m.config.height - inc) * fftSize);

    const auto direction = (m.input.buffer.device_native()) ? cudaMemcpyDeviceToDevice :
                                                            cudaMemcpyHostToDevice;

    auto tensorDataPointer = [](const auto& tensor) -> uint8_t* {
        const auto base = reinterpret_cast<const uint8_t*>(tensor.data());
        return const_cast<uint8_t*>(base) + tensor.offset_bytes();
    };

    uint8_t* bins = tensorDataPointer(frequencyBins);
    const uint8_t* in = tensorDataPointer(m.input.buffer);

    JST_CUDA_CHECK(cudaMemcpyAsync(bins + offset,
                                   in,
                                   size,
                                   direction,
                                   ctx.cuda->stream()), [&]{
        JST_ERROR("Failed to copy data to CUDA device: {}.", err);
    });

    if (size < totalSize) {
        JST_CUDA_CHECK(cudaMemcpyAsync(bins,
                                       in + size,
                                       totalSize - size,
                                       direction,
                                       ctx.cuda->stream()), [&]{
            JST_ERROR("Failed to copy data to CUDA device: {}.", err);
        });
    }

    return Result::SUCCESS;
}

JST_WATERFALL_CUDA(JST_INSTANTIATION)
JST_WATERFALL_CUDA(JST_BENCHMARK)

}  // namespace Jetstream
