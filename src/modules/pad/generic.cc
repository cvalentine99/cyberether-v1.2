#include "jetstream/modules/pad.hh"

namespace Jetstream {

template<Device D, typename T>
Result Pad<D, T>::create() {
    JST_DEBUG("Initializing Pad module.");
    JST_INIT_IO();

    // Check parameters.

    if (config.axis >= input.unpadded.rank()) {
        JST_ERROR("Configuration axis ({}) is larger than the input rank ({}).", config.axis,
                                                                                 input.unpadded.rank());
        return Result::ERROR;
    }

    if (config.offset > config.size) {
        JST_ERROR("Pad offset ({}) must be less than or equal to the configured pad size ({}).",
                  config.offset,
                  config.size);
        return Result::ERROR;
    }

    // Calculate padded shape.

    std::vector<U64> paddedShape = input.unpadded.shape();
    paddedShape[config.axis] += config.size;

    // Allocate output.

    output.padded = Tensor<D, T>(paddedShape);

    return Result::SUCCESS;
}

template<Device D, typename T>
void Pad<D, T>::info() const {
    JST_DEBUG("  Pad Size:   {}", config.size);
    JST_DEBUG("  Pad Axis:   {}", config.axis);
    JST_DEBUG("  Pad Offset: {}", config.offset);
    JST_DEBUG("  Blanking:   {}", config.blank);
}

}  // namespace Jetstream
