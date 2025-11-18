#ifndef JETSTREAM_BACKEND_BASE_HH
#define JETSTREAM_BACKEND_BASE_HH

#include <unordered_map>
#include <variant>
#include <mutex>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/backend/config.hh"

// Backend Architecture:
//
// This file provides a unified interface for managing multiple backend implementations
// (Metal, Vulkan, WebGPU, CPU, CUDA) using compile-time polymorphism.
//
// Key design decisions:
// 1. Macro-based backend registration (JST_BACKEND_REGISTRY) reduces repetition
// 2. Template trait GetBackend<Device> provides type mapping
// 3. std::variant for type-safe heterogeneous storage
// 4. Thread-safe singleton Instance for backend lifecycle management
//
// To add a new backend:
// 1. Create include/jetstream/backend/devices/<name>/base.hh with a class implementing
//    the backend interface (see existing backends for examples)
// 2. Add conditional include below (#ifdef JETSTREAM_BACKEND_<NAME>_AVAILABLE)
// 3. Add GetBackend<Device::<Name>> specialization
// 4. Add std::unique_ptr<<Name>> to BackendHolder variant
// 5. Define JETSTREAM_BACKEND_<NAME>_AVAILABLE in build system (meson.build)

// =============================================================================
// Backend Includes: Conditional compilation based on availability
// =============================================================================

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/devices/metal/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/backend/devices/vulkan/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
#include "jetstream/backend/devices/webgpu/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/backend/devices/cpu/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
#include "jetstream/backend/devices/cuda/base.hh"
#endif

namespace Jetstream::Backend {

// =============================================================================
// Backend Trait: Compile-time type mapping from Device enum to implementation
// =============================================================================

template<Device DeviceId>
struct GetBackend {
    static constexpr bool enabled = false;
    // No Type member = compile error if backend not available
};

// Generate GetBackend specializations for enabled backends
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
template<>
struct GetBackend<Device::Metal> {
    static constexpr bool enabled = true;
    using Type = Metal;
};
#endif

#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
template<>
struct GetBackend<Device::Vulkan> {
    static constexpr bool enabled = true;
    using Type = Vulkan;
};
#endif

#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
template<>
struct GetBackend<Device::WebGPU> {
    static constexpr bool enabled = true;
    using Type = WebGPU;
};
#endif

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
template<>
struct GetBackend<Device::CPU> {
    static constexpr bool enabled = true;
    using Type = CPU;
};
#endif

#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
template<>
struct GetBackend<Device::CUDA> {
    static constexpr bool enabled = true;
    using Type = CUDA;
};
#endif

// =============================================================================
// Backend Instance: Thread-safe singleton managing backend lifecycle
// =============================================================================

class JETSTREAM_API Instance {
 public:
    // Initialize a specific backend with configuration
    template<Device DeviceId>
    Result initialize(const Config& config) {
        static_assert(GetBackend<DeviceId>::enabled,
                     "Backend not available for this device");

        using BackendType = typename GetBackend<DeviceId>::Type;
        std::lock_guard lock(mutex);

        if (!backends.contains(DeviceId)) {
            JST_DEBUG("Initializing {} backend.", DeviceId);
            backends[DeviceId] = std::make_unique<BackendType>(config);
        }
        return Result::SUCCESS;
    }

    // Destroy a specific backend by Device enum
    Result destroy(const Device& id) {
        std::lock_guard lock(mutex);
        if (backends.contains(id)) {
            JST_DEBUG("Destroying {} backend.", id);
            backends.erase(id);
        }
        return Result::SUCCESS;
    }

    // Get backend state (auto-initializes if needed)
    template<Device DeviceId>
    const auto& state() {
        static_assert(GetBackend<DeviceId>::enabled,
                     "Backend not available for this device");

        using BackendType = typename GetBackend<DeviceId>::Type;

        if (!backends.contains(DeviceId)) {
            JST_WARN("The {} backend is not initialized. "
                    "Initializing with default remote settings.", DeviceId);

            Backend::Config config;
            config.remote = true;
            JST_CHECK_THROW(initialize<DeviceId>(config));
        }

        return std::get<std::unique_ptr<BackendType>>(backends[DeviceId]);
    }

    // Check if backend is initialized
    bool initialized(const Device& id) {
        std::lock_guard lock(mutex);
        return backends.contains(id);
    }

    // Destroy all backends
    Result destroyAll() {
        std::lock_guard lock(mutex);
        backends.clear();
        return Result::SUCCESS;
    }

 private:
    // Type-safe heterogeneous container for backend instances
    using BackendHolder = std::variant<
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
        std::unique_ptr<Metal>,
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
        std::unique_ptr<Vulkan>,
#endif
#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
        std::unique_ptr<WebGPU>,
#endif
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
        std::unique_ptr<CPU>,
#endif
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
        std::unique_ptr<CUDA>,
#endif
        std::monostate  // Required for variant to always be valid
    >;

    std::unordered_map<Device, BackendHolder> backends;
    std::mutex mutex;
};

// =============================================================================
// Public API: Global backend management functions
// =============================================================================

// Get the global backend instance singleton
Instance& Get();

// Template-based convenience wrappers

template<Device D>
const auto& State() {
    return Get().state<D>();
}

template<Device D>
Result Initialize(const Config& config) {
    return Get().initialize<D>(config);
}

template<Device D>
Result Destroy() {
    return Get().destroy(D);
}

inline Result Destroy(const Device& id) {
    return Get().destroy(id);
}

template<Device D>
bool Initialized() {
    return Get().initialized(D);
}

inline bool Initialized(const Device& id) {
    return Get().initialized(id);
}

inline Result DestroyAll() {
    return Get().destroyAll();
}

}  // namespace Jetstream::Backend

#endif
