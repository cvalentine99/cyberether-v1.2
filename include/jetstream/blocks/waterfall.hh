#ifndef JETSTREAM_BLOCK_WATERFALL_BASE_HH
#define JETSTREAM_BLOCK_WATERFALL_BASE_HH

#include <limits>

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/waterfall.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Waterfall : public Block {
 public:
    // Configuration

    struct Config {
        F32 zoom = 1.0;
        I32 offset = 0;
        U64 height = 512;
        bool interpolate = true;
        Extent2D<U64> viewSize = {512, 384};

        JST_SERDES(zoom, offset, height, interpolate, viewSize);
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
        JST_SERDES();
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "waterfall";
    }

    std::string name() const {
        return "Waterfall";
    }

    std::string summary() const {
        return "Displays data in a waterfall plot.";
    }

    std::string description() const {
        return "Displays frequency-domain data as a scrolling color-coded waterfall plot with interactive zoom and pan controls. "
               "Takes real-valued spectral data and accumulates it vertically with configurable history depth and optional interpolation. "
               "Supports horizontal zoom, offset adjustment, and smooth interpolation for enhanced visual quality. "
               "Provides intuitive real-time visualization of signal activity, interference patterns, and spectral occupancy in SDR applications.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            waterfall, "waterfall", {
                .zoom = config.zoom,
                .offset = config.offset,
                .height = config.height,
                .interpolate = config.interpolate,
                .viewSize = config.viewSize,
            }, {
                .buffer = input.buffer,
            },
            locale()
        ));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (waterfall) {
            JST_CHECK(instance().eraseModule(waterfall->locale()));
        }

        return Result::SUCCESS;
    }

    // Interface

    void drawPreview(const F32& maxWidth) {
        const auto& size = waterfall->viewSize();
        const auto& ratio = size.ratio();
        const F32 width = (size.x < maxWidth) ? size.x : maxWidth;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((maxWidth - width) / 2.0f));
        ImGui::Image(ImTextureRef(waterfall->getTexture().raw()), ImVec2(width, width/ratio));
    }

    constexpr bool shouldDrawPreview() const {
        return true;
    }

    void drawView() {
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;
        auto [width, height] = waterfall->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(ImTextureRef(waterfall->getTexture().raw()), ImVec2(width/scale.x, height/scale.y));

        if (ImGui::IsItemHovered()) {
            const auto& io = ImGui::GetIO();
            if (io.MouseWheel != 0.0f) {
                auto zoom = waterfall->zoom();
                zoom = std::clamp(zoom + io.MouseWheel * 0.1f, 1.0f, 8.0f);
                waterfall->zoom(zoom);
            }

            const F32 relativeX = getRelativeMousePos().x;
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                if (dragAnchor == std::numeric_limits<I32>::min()) {
                    dragAnchor = static_cast<I32>((relativeX / waterfall->zoom()) + waterfall->offset());
                }
                waterfall->offset(dragAnchor - static_cast<I32>(relativeX / waterfall->zoom()));
            } else {
                dragAnchor = std::numeric_limits<I32>::min();
            }

            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                waterfall->offset(0);
                waterfall->zoom(1.0f);
            }
        } else {
            dragAnchor = std::numeric_limits<I32>::min();
        }
    }

    constexpr bool shouldDrawView() const {
        return true;
    }

    void drawControl() {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Interpolate");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        auto interpolate = waterfall->interpolate();
        if (ImGui::Checkbox("##Interpolate", &interpolate)) {
            waterfall->interpolate(interpolate);
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Zoom");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        auto zoom = waterfall->zoom();
        if (ImGui::DragFloat("##Zoom", &zoom, 0.01, 1.0, 5.0, "%f", ImGuiSliderFlags_AlwaysClamp)) {
            waterfall->zoom(zoom);
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    std::shared_ptr<Jetstream::Waterfall<D, IT>> waterfall;
    I32 dragAnchor = std::numeric_limits<I32>::min();

    ImVec2 getRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Waterfall, is_specialized<Jetstream::Waterfall<D, IT>>::value &&
                            std::is_same<OT, void>::value)

#endif
