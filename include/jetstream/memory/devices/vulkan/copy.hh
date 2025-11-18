#ifndef JETSTREAM_MEMORY_VULKAN_COPY_HH
#define JETSTREAM_MEMORY_VULKAN_COPY_HH

#include <cstddef>
#include <cstring>

#include "jetstream/backend/base.hh"
#include "jetstream/backend/devices/vulkan/helpers.hh"
#include "jetstream/memory/devices/vulkan/tensor.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/tensor.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/memory/devices/metal/tensor.hh"
#endif

namespace Jetstream::Memory {
namespace detail {
inline Result CopyHostToVulkanBuffer(const void* hostPtr,
                                     const VkDeviceSize& byteSize,
                                     const VkDeviceSize& dstOffset,
                                     const VkBuffer& dstBuffer,
                                     const VkDeviceMemory& dstMemory,
                                     const bool& dstHostAccessible) {
    if (byteSize == 0) {
        return Result::SUCCESS;
    }

    auto& backend = Backend::State<Device::Vulkan>();
    if (!backend->isAvailable()) {
        JST_ERROR("[VULKAN:COPY] Vulkan backend is not available.");
        return Result::ERROR;
    }

    auto& device = backend->getDevice();

    if (dstHostAccessible) {
        void* mapped = nullptr;
        JST_VK_CHECK(vkMapMemory(device, dstMemory, 0, dstOffset + byteSize, 0, &mapped), [&]{
            JST_ERROR("[VULKAN:COPY] Failed to map destination memory.");
        });
        std::memcpy(static_cast<std::byte*>(mapped) + dstOffset, hostPtr, byteSize);
        vkUnmapMemory(device, dstMemory);
        return Result::SUCCESS;
    }

    if (byteSize > backend->getStagingBufferSize()) {
        JST_ERROR("[VULKAN:COPY] Copy size ({}) exceeds staging buffer ({}).", byteSize, backend->getStagingBufferSize());
        return Result::ERROR;
    }

    auto* mapped = static_cast<std::byte*>(backend->getStagingBufferMappedMemory());
    std::memcpy(mapped, hostPtr, byteSize);

    return Backend::ExecuteOnce(device,
                                backend->getComputeQueue(),
                                backend->getDefaultFence(),
                                backend->getDefaultCommandBuffer(),
        [&](VkCommandBuffer& commandBuffer) {
            VkBufferCopy region{};
            region.srcOffset = 0;
            region.dstOffset = dstOffset;
            region.size = byteSize;
            vkCmdCopyBuffer(commandBuffer, backend->getStagingBuffer(), dstBuffer, 1, &region);
            return Result::SUCCESS;
        }
    );
}

inline Result CopyVulkanBufferToHost(const VkBuffer& srcBuffer,
                                     const VkDeviceMemory& srcMemory,
                                     const bool& srcHostAccessible,
                                     const VkDeviceSize& byteSize,
                                     const VkDeviceSize& srcOffset,
                                     void* hostPtr) {
    if (byteSize == 0) {
        return Result::SUCCESS;
    }

    auto& backend = Backend::State<Device::Vulkan>();
    if (!backend->isAvailable()) {
        JST_ERROR("[VULKAN:COPY] Vulkan backend is not available.");
        return Result::ERROR;
    }

    auto& device = backend->getDevice();

    if (srcHostAccessible) {
        void* mapped = nullptr;
        JST_VK_CHECK(vkMapMemory(device, srcMemory, 0, srcOffset + byteSize, 0, &mapped), [&]{
            JST_ERROR("[VULKAN:COPY] Failed to map source memory.");
        });
        std::memcpy(hostPtr, static_cast<std::byte*>(mapped) + srcOffset, byteSize);
        vkUnmapMemory(device, srcMemory);
        return Result::SUCCESS;
    }

    if (byteSize > backend->getStagingBufferSize()) {
        JST_ERROR("[VULKAN:COPY] Copy size ({}) exceeds staging buffer ({}).", byteSize, backend->getStagingBufferSize());
        return Result::ERROR;
    }

    auto* mapped = static_cast<std::byte*>(backend->getStagingBufferMappedMemory());

    JST_CHECK(Backend::ExecuteOnce(device,
                                   backend->getComputeQueue(),
                                   backend->getDefaultFence(),
                                   backend->getDefaultCommandBuffer(),
        [&](VkCommandBuffer& commandBuffer) {
            VkBufferCopy region{};
            region.srcOffset = srcOffset;
            region.dstOffset = 0;
            region.size = byteSize;
            vkCmdCopyBuffer(commandBuffer, srcBuffer, backend->getStagingBuffer(), 1, &region);
            return Result::SUCCESS;
        }
    ));

    std::memcpy(hostPtr, mapped, byteSize);
    return Result::SUCCESS;
}
}  // namespace detail

template<typename T>
inline Result Copy(Tensor<Device::Vulkan, T>& dst, const Tensor<Device::Vulkan, T>& src) {
    if ((dst.size() != src.size()) ||
        (dst.shape() != src.shape()) ||
        (!dst.contiguous() || !src.contiguous())) {
        JST_ERROR("[VULKAN:COPY] Copy not implemented for non-contiguous or mismatched tensors.");
        return Result::ERROR;
    }

    auto& backend = Backend::State<Device::Vulkan>();
    if (!backend->isAvailable()) {
        JST_ERROR("[VULKAN:COPY] Vulkan backend is not available.");
        return Result::ERROR;
    }

    return Backend::ExecuteOnce(backend->getDevice(),
                                backend->getComputeQueue(),
                                backend->getDefaultFence(),
                                backend->getDefaultCommandBuffer(),
        [&](VkCommandBuffer& commandBuffer) {
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = src.offset_bytes();
            copyRegion.dstOffset = dst.offset_bytes();
            copyRegion.size = dst.size_bytes();
            vkCmdCopyBuffer(commandBuffer, src.data(), dst.data(), 1, &copyRegion);
            return Result::SUCCESS;
        }
    );
}

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
template<typename T>
inline Result Copy(Tensor<Device::Vulkan, T>& dst, const Tensor<Device::CPU, T>& src) {
    if ((dst.size() != src.size()) ||
        (dst.shape() != src.shape()) ||
        (!dst.contiguous() || !src.contiguous())) {
        JST_ERROR("[VULKAN:COPY] Copy not implemented for non-contiguous or mismatched tensors.");
        return Result::ERROR;
    }

    const void* srcPtr = static_cast<const void*>(src.data() + src.offset());
    return detail::CopyHostToVulkanBuffer(srcPtr,
                                          dst.size_bytes(),
                                          dst.offset_bytes(),
                                          dst.data(),
                                          dst.memory(),
                                          dst.host_accessible());
}

template<typename T>
inline Result Copy(Tensor<Device::CPU, T>& dst, const Tensor<Device::Vulkan, T>& src) {
    if ((dst.size() != src.size()) ||
        (dst.shape() != src.shape()) ||
        (!dst.contiguous() || !src.contiguous())) {
        JST_ERROR("[VULKAN:COPY] Copy not implemented for non-contiguous or mismatched tensors.");
        return Result::ERROR;
    }

    void* dstPtr = static_cast<void*>(dst.data() + dst.offset());
    return detail::CopyVulkanBufferToHost(src.data(),
                                          src.memory(),
                                          src.host_accessible(),
                                          src.size_bytes(),
                                          src.offset_bytes(),
                                          dstPtr);
}
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
template<typename T>
inline Result Copy(Tensor<Device::Vulkan, T>& dst, const Tensor<Device::Metal, T>& src) {
    if ((dst.size() != src.size()) ||
        (dst.shape() != src.shape()) ||
        (!dst.contiguous() || !src.contiguous())) {
        JST_ERROR("[VULKAN:COPY] Copy not implemented for non-contiguous or mismatched tensors.");
        return Result::ERROR;
    }

    const auto* srcPtr = static_cast<const std::byte*>(src.data()->contents()) + src.offset_bytes();
    return detail::CopyHostToVulkanBuffer(srcPtr,
                                          dst.size_bytes(),
                                          dst.offset_bytes(),
                                          dst.data(),
                                          dst.memory(),
                                          dst.host_accessible());
}

template<typename T>
inline Result Copy(Tensor<Device::Metal, T>& dst, const Tensor<Device::Vulkan, T>& src) {
    if ((dst.size() != src.size()) ||
        (dst.shape() != src.shape()) ||
        (!dst.contiguous() || !src.contiguous())) {
        JST_ERROR("[VULKAN:COPY] Copy not implemented for non-contiguous or mismatched tensors.");
        return Result::ERROR;
    }

    auto* dstPtr = static_cast<std::byte*>(dst.data()->contents()) + dst.offset_bytes();
    auto result = detail::CopyVulkanBufferToHost(src.data(),
                                                 src.memory(),
                                                 src.host_accessible(),
                                                 src.size_bytes(),
                                                 src.offset_bytes(),
                                                 dstPtr);
    if (result == Result::SUCCESS) {
        dst.data()->didModifyRange(NS::Range(dst.offset_bytes(), dst.offset_bytes() + dst.size_bytes()));
    }
    return result;
}
#endif

}  // namespace Jetstream::Memory

#endif
