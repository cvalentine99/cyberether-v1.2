#include "jetstream/render/devices/metal/buffer.hh"

namespace Jetstream::Render {

using Implementation = BufferImp<Device::Metal>;

Implementation::BufferImp(const Config& config) : Buffer(config) {
}

Result Implementation::create() {
    JST_DEBUG("[METAL] Creating buffer.");

    auto device = Backend::State<Device::Metal>()->getDevice();
    const auto& byteSize = config.size * config.elementByteSize;

    if (config.enableZeroCopy) {
        buffer = static_cast<MTL::Buffer*>(config.buffer);
        buffer->retain();
    } else {
        // Set appropriate resource options based on buffer usage.
        MTL::ResourceOptions resourceOptions = MTL::ResourceStorageModeShared;

        // Use write-combined CPU cache mode for buffers that are frequently updated from CPU.
        // This improves performance for vertex, index, and uniform buffers that change each frame.
        if ((config.target & Target::VERTEX) == Target::VERTEX ||
            (config.target & Target::VERTEX_INDICES) == Target::VERTEX_INDICES ||
            (config.target & Target::UNIFORM) == Target::UNIFORM ||
            (config.target & Target::UNIFORM_DYNAMIC) == Target::UNIFORM_DYNAMIC) {
            resourceOptions |= MTL::ResourceCPUCacheModeWriteCombined;
        }

        // Storage buffers can benefit from default cache mode if read back to CPU,
        // or private storage if GPU-only (though we use shared for CPU accessibility).
        // Keep default cache mode for storage buffers to allow efficient CPU readback.

        buffer = device->newBuffer(config.buffer,
                                   byteSize,
                                   resourceOptions);
    }
    JST_ASSERT(buffer, "Failed to create buffer.");

    return Result::SUCCESS;
}

Result Implementation::destroy() {
    JST_DEBUG("[METAL] Destroying buffer.");

    if (buffer) {
        buffer->release();
    }
    buffer = nullptr;

    return Result::SUCCESS;
}

Result Implementation::update() {
    return update(0, config.size);
}

Result Implementation::update(const U64& offset, const U64& size) {
    if (size == 0 || config.enableZeroCopy) {
        return Result::SUCCESS;
    }

    const auto& byteOffset = offset * config.elementByteSize;
    const auto& byteSize = size * config.elementByteSize;


    uint8_t* ptr = static_cast<uint8_t*>(buffer->contents());
    memcpy(ptr + byteOffset, (uint8_t*)config.buffer + byteOffset, byteSize);
#if !defined(TARGET_OS_IOS)
    buffer->didModifyRange(NS::Range(byteOffset, byteSize));
#endif

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render
