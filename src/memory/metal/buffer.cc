#include "jetstream/memory/devices/metal/buffer.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/buffer.hh"
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/memory/devices/vulkan/buffer.hh"
#include "jetstream/memory/devices/vulkan/copy.hh"
#endif

#include <cstddef>

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/memory/devices/cuda/buffer.hh"
#endif

namespace Jetstream {

using Implementation = TensorBuffer<Device::Metal>;

Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>& storage,
                             const TensorPrototypeMetadata& prototype) {
    JST_TRACE("[METAL:BUFFER] Allocating new buffer.");

    // Check if Metal is available.

    if (!Backend::State<Device::Metal>()->isAvailable()) {
        JST_TRACE("[METAL:BUFFER] Metal is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Initialize storage.

    storage->root_device = Device::Metal;
    storage->compatible_devices = {
        Device::Metal
    };

    // Allocate memory buffer.

    if (prototype.size_bytes > 0) {
        auto device = Backend::State<Device::Metal>()->getDevice();
        const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);
        buffer = device->newBuffer(alignedSizeBytes, MTL::ResourceStorageModeShared);
        if (!buffer) {
            JST_ERROR("[METAL:BUFFER] Failed to allocate memory.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Set buffer flags.

        set_allocated();
        set_host_accessible();
        set_device_native();
        set_host_native();

        // Null out array.

        memset(buffer->contents(), 0, prototype.size_bytes);
    }

    // Add compatible devices.

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    if (TensorBuffer<Device::CPU>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::CPU);
    }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
    if (TensorBuffer<Device::Vulkan>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::Vulkan);
    }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
    if (TensorBuffer<Device::CUDA>::CanImport(*this)) {
        storage->compatible_devices.insert(Device::CUDA);
    }
#endif
}

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::CPU>>& root_buffer) {
    JST_TRACE("[METAL:BUFFER] Cloning from CPU buffer.");

    // Check if Metal is available.

    if (!Backend::State<Device::Metal>()->isAvailable()) {
        JST_TRACE("[METAL:BUFFER] Metal is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check if root buffer can be imported.

    if (!TensorBuffer<Device::Metal>::CanImport(*root_buffer)) {
        JST_TRACE("[METAL:BUFFER] CPU buffer is not compatible with Metal.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Check size.

    if (prototype.size_bytes == 0) {
        return;
    }

    // Check alignment.

    if (!JST_IS_ALIGNED(root_buffer->data()) && root_buffer->data() != nullptr) {
        JST_TRACE("[METAL:BUFFER] Buffer is not aligned. Cannot clone CPU memory.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Allocate memory buffer.

    auto device = Backend::State<Device::Metal>()->getDevice();
    auto* ptr = root_buffer->data();
    const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);
    buffer = device->newBuffer(ptr, alignedSizeBytes, MTL::ResourceStorageModeShared, nullptr);
    if (!buffer) {
        JST_ERROR("[METAL:BUFFER] Failed to allocate memory.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Set buffer flags.

    set_host_accessible();
    set_device_native();
    set_host_native();
}

bool Implementation::CanImport(const TensorBuffer<Device::CPU>& root_buffer) noexcept {
    JST_TRACE("[METAL:BUFFER] Checking if CPU buffer can be imported.");

    // Allow importing empty buffers.

    if (!root_buffer.allocated()) {
        return true;
    }

    // Check if Metal is available.

    if (!Backend::State<Device::Metal>()->isAvailable()) {
        JST_TRACE("[METAL:BUFFER] Metal is not available.");
        return false;
    }

    // Check if Metal buffer is host accessible.

    if (!Backend::State<Device::Metal>()->hasUnifiedMemory()) {
        JST_TRACE("[METAL:BUFFER] Metal buffer is not unified.");
        return false;
    }

    // Check if Metal buffer size is aligned.

    if (!JST_IS_ALIGNED(root_buffer.data())) {
        JST_TRACE("[METAL:BUFFER] Buffer is not aligned. Cannot import CPU memory.");
        return false;
    }

    return true;
}
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata& prototype,
                             const std::shared_ptr<TensorBuffer<Device::Vulkan>>& root_buffer) {
    JST_TRACE("[METAL:BUFFER] Importing Vulkan buffer.");

    if (!Backend::State<Device::Metal>()->isAvailable()) {
        JST_TRACE("[METAL:BUFFER] Metal is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    if (!Backend::State<Device::Vulkan>()->isAvailable()) {
        JST_TRACE("[METAL:BUFFER] Vulkan is not available.");
        JST_CHECK_THROW(Result::ERROR);
    }

    if (!TensorBuffer<Device::Metal>::CanImport(*root_buffer)) {
        JST_TRACE("[METAL:BUFFER] Vulkan buffer is not compatible with Metal.");
        JST_CHECK_THROW(Result::ERROR);
    }

    if (prototype.size_bytes == 0 || !root_buffer->allocated()) {
        return;
    }

    auto device = Backend::State<Device::Metal>()->getDevice();
    const auto alignedSizeBytes = JST_PAGE_ALIGNED_SIZE(prototype.size_bytes);
    buffer = device->newBuffer(alignedSizeBytes, MTL::ResourceStorageModeShared);
    if (!buffer) {
        JST_ERROR("[METAL:BUFFER] Failed to allocate memory.");
        JST_CHECK_THROW(Result::ERROR);
    }

    set_allocated();
    set_host_accessible();
    set_device_native();
    set_host_native();

    auto* dstPtr = static_cast<std::byte*>(buffer->contents()) + prototype.offset_bytes;
    JST_CHECK_THROW(Memory::detail::CopyVulkanBufferToHost(root_buffer->data(),
                                                          root_buffer->memory(),
                                                          root_buffer->host_accessible(),
                                                          prototype.size_bytes,
                                                          prototype.offset_bytes,
                                                          dstPtr));

    buffer->didModifyRange(NS::Range(prototype.offset_bytes, prototype.offset_bytes + prototype.size_bytes));
}

bool Implementation::CanImport(const TensorBuffer<Device::Vulkan>&) noexcept {
    if (!Backend::State<Device::Metal>()->isAvailable()) {
        return false;
    }
    if (!Backend::State<Device::Vulkan>()->isAvailable()) {
        return false;
    }
    return true;
}
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
Implementation::TensorBuffer(std::shared_ptr<TensorStorageMetadata>&,
                             const TensorPrototypeMetadata&,
                             const std::shared_ptr<TensorBuffer<Device::CUDA>>&) {
    throw std::runtime_error("CUDA buffers are not supported on Metal.");
}

bool Implementation::CanImport(const TensorBuffer<Device::CUDA>&) noexcept {
    return false;
}
#endif

Implementation::~TensorBuffer() {
    JST_TRACE("[METAL:BUFFER] Releasing buffer {}.", jst::fmt::ptr(buffer));

    if (buffer) {
        buffer->release();
    }
}

}  // namespace Jetstream
