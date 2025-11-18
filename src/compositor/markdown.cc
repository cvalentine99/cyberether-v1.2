#include "jetstream/compositor.hh"
#include "jetstream/instance.hh"
#include "jetstream/platform.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

namespace Jetstream {

void Compositor::ImGuiMarkdownStyleSetup() {
    styling.config.linkCallback        = &Compositor::ImGuiMarkdownLinkCallback;
    styling.config.tooltipCallback     = nullptr;
    styling.config.imageCallback       = nullptr;
    styling.config.linkIcon            = ICON_FA_LINK;
    styling.config.headingFormats[0]   = { styling.fonts.h1, true };
    styling.config.headingFormats[1]   = { styling.fonts.h2, true };
    styling.config.headingFormats[2]   = { styling.fonts.bold, false };
    styling.config.userData            = this;
    styling.config.formatCallback      = &Compositor::ImGuiMarkdownFormatCallback;
}

void Compositor::ImGuiMarkdownLinkCallback(ImGui::MarkdownLinkCallbackData data) {
    std::string url(data.link, data.linkLength);
    if(!data.isImage) {
        Platform::OpenUrl(url);
    }
}

void Compositor::ImGuiMarkdownFormatCallback(const ImGui::MarkdownFormatInfo& md_info, bool start) {
    switch (md_info.type) {
        case ImGui::MarkdownFormatType::NORMAL_TEXT:
            break;
        case ImGui::MarkdownFormatType::EMPHASIS: {
            ImGui::MarkdownHeadingFormat fmt;
            fmt = md_info.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
            if (start) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (fmt.font) {
                    ImGui::PushFont(fmt.font, 0.0f);
                }
            } else {
                if (fmt.font) {
                    ImGui::PopFont();
                }
                ImGui::PopStyleColor();
            }
            break;
        }
        case ImGui::MarkdownFormatType::HEADING: {
            ImGui::MarkdownHeadingFormat fmt;
            if (md_info.level > ImGui::MarkdownConfig::NUMHEADINGS) {
                fmt = md_info.config->headingFormats[ImGui::MarkdownConfig::NUMHEADINGS - 1];
            } else {
                fmt = md_info.config->headingFormats[md_info.level - 1];
            }
            if (start) {
                if (fmt.font) {
                    ImGui::PushFont(fmt.font, 0.0f);
                }
            } else {
                if (fmt.separator) {
                    ImGui::Separator();
                }
                if (fmt.font) {
                    ImGui::PopFont();
                }
            }
            break;
        }
        case ImGui::MarkdownFormatType::UNORDERED_LIST:
            break;
        case ImGui::MarkdownFormatType::LINK: {
            if (start) {
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(71, 127, 239, 255));
            } else {
                ImGui::PopStyleColor();
                if (md_info.itemHovered) {
                    ImGui::UnderLine(IM_COL32(71, 127, 239, 255));
                }
            }
            break;
        }
    }
}

}  // namespace Jetstream
