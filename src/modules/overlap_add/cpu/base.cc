#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct OverlapAdd<D, T>::Impl {
    Tensor<D, T> previousOverlap;
};

template<Device D, typename T>
OverlapAdd<D, T>::OverlapAdd() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
OverlapAdd<D, T>::~OverlapAdd() {
    impl.reset();
}

template<Device D, typename T>
Result OverlapAdd<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Overlap Add compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result OverlapAdd<D, T>::compute(const Context&) {
    const auto axis = config.axis;
    const auto rank = input.buffer.rank();

    std::vector<U64> coord(rank, 0);
    std::vector<U64> overlapCoord(rank, 0);
    std::vector<U64> prevCoord(rank, 0);

    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = input.buffer[i];
    }

    const auto& overlapShape = input.overlap.shape();
    const auto& prevShape = impl->previousOverlap.shape();

    auto clampBroadcast = [](std::vector<U64>& target, const std::vector<U64>& sourceShape) {
        for (U64 dim = 0; dim < sourceShape.size(); ++dim) {
            if (sourceShape[dim] == 1) {
                target[dim] = 0;
            }
        }
    };

    for (U64 i = 0; i < output.buffer.size(); ++i) {
        output.buffer.offset_to_shape(i, coord);
        auto& sample = output.buffer[coord];

        if (coord[axis] == 0) {
            prevCoord = coord;
            prevCoord[axis] = 0;
            clampBroadcast(prevCoord, prevShape);
            sample += impl->previousOverlap[prevCoord];
        } else {
            overlapCoord = coord;
            overlapCoord[axis] -= 1;
            clampBroadcast(overlapCoord, overlapShape);
            sample += input.overlap[overlapCoord];
        }
    }

    const U64 lastSlice = (overlapShape[axis] > 0) ? overlapShape[axis] - 1 : 0;
    std::vector<U64> writeCoord(rank, 0);
    std::vector<U64> sourceCoord(rank, 0);

    for (U64 i = 0; i < impl->previousOverlap.size(); ++i) {
        impl->previousOverlap.offset_to_shape(i, writeCoord);
        sourceCoord = writeCoord;
        sourceCoord[axis] = lastSlice;
        clampBroadcast(sourceCoord, overlapShape);
        impl->previousOverlap[i] = input.overlap[sourceCoord];
    }

    return Result::SUCCESS;
}

JST_OVERLAP_ADD_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
