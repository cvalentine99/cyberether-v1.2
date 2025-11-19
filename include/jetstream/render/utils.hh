#ifndef JETSTREAM_RENDER_UTILS_HH
#define JETSTREAM_RENDER_UTILS_HH

#include "jetstream/render/base.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/types.hh"

namespace Jetstream {

template<Device D, typename T>
inline std::tuple<void*, bool> ConvertToOptimalStorage(auto& window, Tensor<D, T>& tensor) {
    void* buffer = nullptr;
    bool enableZeroCopy = false;

    Device renderDevice = window->device();
    Device optimalDevice = Device::None;

    (void)optimalDevice;
    (void)renderDevice;

    if (tensor.device() == renderDevice) {
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        if (renderDevice == Device::CPU) {
            buffer = tensor.data();
            enableZeroCopy = tensor.device_native();
            optimalDevice = Device::CPU;
        }
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        if (renderDevice == Device::Metal) {
            buffer = tensor.data();
            enableZeroCopy = tensor.device_native();
            optimalDevice = Device::Metal;
        }
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        if (renderDevice == Device::Vulkan) {
            buffer = tensor.data();
            enableZeroCopy = tensor.device_native();
            optimalDevice = Device::Vulkan;
        }
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        if (renderDevice == Device::CUDA) {
            buffer = tensor.data();
            enableZeroCopy = tensor.device_native();
            optimalDevice = Device::CUDA;
        }
#endif
    } else {
        auto& optimal = MapOn<Device::CPU>(tensor);
        buffer = optimal.data();
        enableZeroCopy = false;
        optimalDevice = Device::CPU;
    }

    JST_TRACE("[RENDER] Tensor Device: {} | Render Device: {} | Optimal Device: {} | Zero-Copy: {}",
              tensor.device(), renderDevice, optimalDevice, enableZeroCopy ? "YES" : "NO");

    return {buffer, enableZeroCopy};
}

}  // namespace Jetstream

#endif
