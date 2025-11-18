#include "jetstream/modules/overlap_add.hh"

namespace Jetstream {

template<Device D, typename T>
Result OverlapAdd<D, T>::create() {
    JST_DEBUG("Initializing Overlap Add module.");
    JST_INIT_IO();

    // Check parameters.

    if (config.axis >= input.buffer.rank()) {
        JST_ERROR("Configuration axis ({}) is larger than the input rank ({}).", config.axis,
                                                                                 input.buffer.rank());
        return Result::ERROR;
    }

    if (input.buffer.rank() != input.overlap.rank()) {
        JST_ERROR("Input buffer rank ({}) is not equal to the overlap rank ({}).",
                  input.buffer.rank(), input.overlap.rank());
        return Result::ERROR;
    }

    const auto& bufferShape = input.buffer.shape();
    const auto& overlapShape = input.overlap.shape();

    if (bufferShape[config.axis] != overlapShape[config.axis]) {
        JST_ERROR("Overlap axis dimension ({}) must match buffer axis dimension ({}).",
                  overlapShape[config.axis],
                  bufferShape[config.axis]);
        return Result::ERROR;
    }

    for (U64 dim = 0; dim < bufferShape.size(); ++dim) {
        if (dim == config.axis) {
            continue;
        }

        if (bufferShape[dim] == overlapShape[dim]) {
            continue;
        }

        if (overlapShape[dim] != 1) {
            JST_ERROR("Cannot broadcast overlap dimension {} (size {}) to buffer dimension {} (size {}).",
                      dim,
                      overlapShape[dim],
                      dim,
                      bufferShape[dim]);
            return Result::ERROR;
        }
    }

    // Allocate output.

    output.buffer = Tensor<D, T>(bufferShape);

    auto previousOverlapShape = overlapShape;
    previousOverlapShape[config.axis] = 1;
    impl->previousOverlap = Tensor<D, T>(previousOverlapShape);

    return Result::SUCCESS;
}

template<Device D, typename T>
void OverlapAdd<D, T>::info() const {
    JST_DEBUG("  Axis:   {}", config.axis);
}

}  // namespace Jetstream
