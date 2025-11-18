#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>
#include <thread>

#include "jetstream/base.hh"

using namespace Jetstream;

#ifdef JST_OS_BROWSER
extern "C" {
EMSCRIPTEN_KEEPALIVE
void cyberether_shutdown() {
    JST_INFO("Shutting down...");
    emscripten_cancel_main_loop();
    emscripten_runtime_keepalive_pop();
    emscripten_force_exit(0);
}
}
#endif

int main(int argc, char* argv[]) {
    // Parse command line arguments.

    Backend::Config backendConfig;
    Viewport::Config viewportConfig;
    Render::Window::Config renderConfig;
    std::string flowgraphPath;
    Device prefferedBackend = Device::None;

    bool validationExplicitlyDisabled = false;

    for (int i = 1; i < argc; i++) {
        const std::string arg = std::string(argv[i]);

        if (arg == "--remote") {
            backendConfig.remote = true;

            continue;
        }

        if (arg == "--auto-join") {
            viewportConfig.autoJoin = true;

            continue;
        }

        if (arg == "--broker") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --broker requires a URL." << std::endl;
                return 1;
            }
            viewportConfig.broker = argv[++i];

            continue;
        }

        if (arg == "--backend") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --backend requires a value." << std::endl;
                return 1;
            }

            const std::string backendName = argv[++i];
            try {
                prefferedBackend = StringToDevice(backendName);
            } catch (const Device&) {
                std::cerr << "[ERROR] Unknown backend '" << backendName
                          << "'. Valid options: Metal, Vulkan, WebGPU, CPU, CUDA, None." << std::endl;
                return 1;
            }

            continue;
        }

        if (arg == "--no-validation") {
            backendConfig.validationEnabled = false;
            validationExplicitlyDisabled = true;

            continue;
        }

        if (arg == "--no-vsync") {
            viewportConfig.vsync = false;

            continue;
        }

        if (arg == "--no-hw-acceleration") {
            viewportConfig.hardwareAcceleration = false;

            continue;
        }

        if (arg == "--benchmark") {
            std::string outputType = "markdown";

            if (i + 1 < argc) {
                outputType = std::string(argv[++i]);
            }

            std::string normalized = outputType;
            std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            static const std::array<std::string_view, 4> kValidBenchmarkOutputs = {
                "markdown", "json", "csv", "quiet"
            };

            const auto isValid = std::any_of(kValidBenchmarkOutputs.begin(),
                                             kValidBenchmarkOutputs.end(),
                                             [&](std::string_view candidate) {
                                                 return normalized == candidate;
                                             });
            if (!isValid) {
                std::cerr << "[ERROR] Unknown benchmark output '" << outputType
                          << "'. Valid options: markdown, json, csv, quiet." << std::endl;
                return 1;
            }

            Benchmark::Run(normalized);

            return 0;
        }

        if (arg == "--framerate") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --framerate requires a value." << std::endl;
                return 1;
            }
            viewportConfig.framerate = std::stoul(argv[++i]);

            continue;
        }

        if (arg == "--multisampling") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --multisampling requires a value." << std::endl;
                return 1;
            }
            backendConfig.multisampling = std::stoul(argv[++i]);

            continue;
        }

        if (arg == "--size") {
            if (i + 2 >= argc) {
                std::cerr << "[ERROR] --size requires width and height values." << std::endl;
                return 1;
            }
            viewportConfig.size.x = std::stoul(argv[++i]);
            viewportConfig.size.y = std::stoul(argv[++i]);

            continue;
        }

        if (arg == "--codec") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --codec requires a value." << std::endl;
                return 1;
            }

            const std::string codecName = argv[++i];
            try {
                viewportConfig.codec = Viewport::StringToVideoCodec(codecName);
            } catch (const Result&) {
                std::cerr << "[ERROR] Unknown codec '" << codecName
                          << "'. Valid options: H264, AV1, VP8, VP9, FFV1." << std::endl;
                return 1;
            }

            continue;
        }

        if (arg == "--device-id") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --device-id requires a numeric value." << std::endl;
                return 1;
            }
            backendConfig.deviceId = std::stoul(argv[++i]);

            continue;
        }

        if (arg == "--staging-buffer") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --staging-buffer requires a size in megabytes." << std::endl;
                return 1;
            }
            backendConfig.stagingBufferSize = std::stoul(argv[++i]) * 1024 * 1024;

            continue;
        }

        if (arg == "--scale") {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] --scale requires a numeric value." << std::endl;
                return 1;
            }
            renderConfig.scale = std::stof(argv[++i]);

            continue;
        }

        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] [flowgraph]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --remote                Enable remote viewport mode." << std::endl;
            std::cout << "  --auto-join             Automatically approve remote sessions (insecure)." << std::endl;
            std::cout << "  --broker [url]          Set the broker of the remote viewport. Default: `https://api.cyberether.org`" << std::endl;
            std::cout << "  --backend [backend]     Set the preferred backend (`Metal`, `Vulkan`, or `WebGPU`)." << std::endl;
            std::cout << "  --framerate [value]     Set the framerate of the te viewport (FPS). Default: `60`" << std::endl;
            std::cout << "  --multisampling [value] Set the multisampling anti-aliasing factor (`1`, `4`, or `8`). Default: `4`" << std::endl;
            std::cout << "  --codec [codec]         Set the video codec of the remote viewport. Default: `H264`" << std::endl;
            std::cout << "  --size [width] [height] Set the initial size of the viewport. Default: `1920 1080`" << std::endl;
            std::cout << "  --scale [scale]         Set the scale of the render window. Default: `1.0`" << std::endl;
            std::cout << "  --benchmark [type]      Run the benchmark and output the results (`markdown`, `json`, or `csv`). Default: `markdown`" << std::endl;
            std::cout << "  --no-hw-acceleration    Disable hardware acceleration. Enabled otherwise." << std::endl;
            std::cout << "Other Options:" << std::endl;
            std::cout << "  --staging-buffer [size] Set the staging buffer size (MB). Default: `64`" << std::endl;
            std::cout << "  --device-id [id]        Set the physical device ID. Default: `0`" << std::endl;
            std::cout << "  --no-validation         Disable Vulkan validation layers. Enabled otherwise." << std::endl;
            std::cout << "  --no-vsync              Disable vsync. Enabled otherwise." << std::endl;
            std::cout << "Other:" << std::endl;
            std::cout << "  --help, -h              Print this help message." << std::endl;
            std::cout << "  --version, -v           Print the version." << std::endl;
            return 0;
        }

        if (arg == "--version" || arg == "-v") {
            std::cout << "CyberEther v" << JETSTREAM_VERSION_STR << "-" << JETSTREAM_BUILD_TYPE << std::endl;
            return 0;
        }

        flowgraphPath = arg;
    }

    if (validationExplicitlyDisabled &&
        (prefferedBackend != Device::Vulkan && prefferedBackend != Device::None)) {
        std::cerr << "[WARN] --no-validation currently only affects Vulkan backends. "
                  << "Validation will remain enabled for the selected backend." << std::endl;
        backendConfig.validationEnabled = true;
    }

    // Instance creation.

    Instance instance;

    // Configure instance.

    Instance::Config config = {
        .preferredDevice = prefferedBackend,
        .renderCompositor = true,
        .backendConfig = backendConfig,
        .viewportConfig = viewportConfig,
        .renderConfig = renderConfig
    };

    JST_CHECK_THROW(instance.build(config));

    // Load flowgraph if provided.

    if (!flowgraphPath.empty()) {
        JST_CHECK_THROW(instance.flowgraph().create());
        JST_CHECK_THROW(instance.flowgraph().importFromFile(flowgraphPath));
    }

    // Start instance.

    instance.start();

    // Start compute thread.

    auto computeThread = std::thread([&]{
        while (instance.computing()) {
            JST_CHECK_THROW(instance.compute());
        }
    });

    // Start graphical thread.

    auto graphicalThreadLoop = [](void* arg) {
        Instance* instance = reinterpret_cast<Instance*>(arg);

        if (instance->begin() == Result::SKIP) {
            return;
        }
        JST_CHECK_THROW(instance->present());
        if (instance->end() == Result::SKIP) {
            return;
        }
    };

#ifdef JST_OS_BROWSER
    emscripten_set_main_loop_arg(graphicalThreadLoop, &instance, 0, 1);
#else
    auto graphicalThread = std::thread([&]{
        while (instance.presenting()) {
            graphicalThreadLoop(&instance);
        }
    });
#endif

    // Start input polling.

#ifdef JST_OS_BROWSER
    emscripten_runtime_keepalive_push();
#else
    while (instance.running()) {
        instance.viewport().waitEvents();
    }
#endif

    // Stop instance and wait for threads.

    instance.reset();
    instance.stop();

    if (computeThread.joinable()) {
        computeThread.join();
    }

#ifdef JST_OS_BROWSER
    emscripten_cancel_main_loop();
#else
    if (graphicalThread.joinable()) {
        graphicalThread.join();
    }
#endif

    // Destroy instance.

    instance.destroy();

    // Destroy backend.

    Backend::DestroyAll();

    return 0;
}
