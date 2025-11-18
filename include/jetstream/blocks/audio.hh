#ifndef JETSTREAM_BLOCK_AUDIO_BASE_HH
#define JETSTREAM_BLOCK_AUDIO_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/audio.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Audio : public Block {
 public:
    using AudioModule = Jetstream::Audio<D, IT>;

    // Configuration

    struct Config {
        std::string deviceName = "Default";
        F32 inSampleRate = 48e3;
        F32 outSampleRate = 48e3;
        U32 channels = 1;

        JST_SERDES(deviceName, inSampleRate, outSampleRate, channels);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "audio";
    }

    std::string name() const {
        return "Audio";
    }

    std::string summary() const {
        return "Plays input on the speaker.";
    }

    std::string description() const {
        return "Routes audio data to a system playback device with sample rate conversion. Takes a real-valued "
               "input tensor and resamples it from the input sample rate to the output sample rate before playback. "
               "Supports device selection and automatic rate matching. Internally uses a linear resampler and "
               "circular buffer for smooth playback on the selected audio device.";
    }

    // Constructor

    Result create() {
        // Populate internal state.

        availableDeviceList = AudioModule::ListAvailableDevices();

        // Starting audio module.

        JST_CHECK(instance().addModule(
            audio, "audio", {
                .deviceName = config.deviceName,
                .inSampleRate = config.inSampleRate,
                .outSampleRate = config.outSampleRate,
                .channels = config.channels,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));
        JST_CHECK(Block::LinkOutput("buffer", output.buffer, audio->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (audio) {
            JST_CHECK(instance().eraseModule(audio->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Sample Rate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        F32 inSampleRate = config.inSampleRate / 1e6f;
        if (ImGui::InputFloat("##in-sample-rate", &inSampleRate, 0.1f, 0.2f, "%.3f MHz", ImGuiInputTextFlags_EnterReturnsTrue)) {
            config.inSampleRate = inSampleRate * 1e6;

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Channels");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        int channels = static_cast<int>(config.channels);
        if (ImGui::InputInt("##channels", &channels, 1, 2, ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (channels < 1) channels = 1;
            if (channels > 8) channels = 8;  // Reasonable limit for most audio devices
            config.channels = static_cast<U32>(channels);

            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
            });
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Device List");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        static const char* noDeviceMessage = "No device found";
        if (ImGui::BeginCombo("##DeviceList", availableDeviceList.empty() ? noDeviceMessage : audio->getDeviceName().c_str())) {
            for (const auto& device : availableDeviceList) {
                bool isSelected = (config.deviceName == device);
                if (ImGui::Selectable(device.c_str(), isSelected)) {
                    config.deviceName = device;

                    JST_DISPATCH_ASYNC([&](){
                        ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });
                        JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                    });
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();

                }
            }
            ImGui::EndCombo();
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TableSetColumnIndex(1);
        const F32 fullWidth = ImGui::GetContentRegionAvail().x;
        if (ImGui::Button("Reload Device List", ImVec2(fullWidth, 0))) {
            JST_DISPATCH_ASYNC([&](){
                ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading device list..." });
                availableDeviceList = AudioModule::ListAvailableDevices();
                JST_CHECK_NOTIFY(Result::SUCCESS);
            });
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<AudioModule> audio;

    typename AudioModule::DeviceList availableDeviceList;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Audio, is_specialized<Jetstream::Audio<D, IT>>::value &&
                        std::is_same<OT, void>::value)

#endif
