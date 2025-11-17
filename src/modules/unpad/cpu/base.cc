#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct Unpad<D, T>::Impl {};

template<Device D, typename T>
Unpad<D, T>::Unpad() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Unpad<D, T>::~Unpad() {
    impl.reset();
}

template<Device D, typename T>
Result Unpad<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Unpad compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Unpad<D, T>::compute(const Context&) {
    std::vector<U64> shape = input.padded.shape();
    const U64 axis = config.axis;
    const U64 totalAxisSize = shape[axis];
    const U64 dataAxisSize = totalAxisSize - config.size;
    const U64 frontPad = config.offset;
    const U64 tailPadStart = frontPad + dataAxisSize;

    for (U64 i = 0; i < input.padded.size(); i++) {
        input.padded.offset_to_shape(i, shape);
        const U64 axisIndex = shape[axis];

        if (axisIndex < frontPad) {
            output.pad[shape] = input.padded[i];
            continue;
        }

        if (axisIndex >= tailPadStart) {
            shape[axis] -= dataAxisSize;
            output.pad[shape] = input.padded[i];
            continue;
        }

        shape[axis] -= frontPad;
        output.unpadded[shape] = input.padded[i];
    }

    return Result::SUCCESS;
}

JST_UNPAD_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
