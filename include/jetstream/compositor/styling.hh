#ifndef JETSTREAM_COMPOSITOR_STYLING_HH
#define JETSTREAM_COMPOSITOR_STYLING_HH

#include "jetstream/render/tools/imgui.h"
#include "jetstream/render/tools/imgui_markdown.hh"
#include "jetstream/types.hh"

namespace Jetstream::CompositorStyling {

// =============================================================================
// Device-Specific Color Constants
// =============================================================================
// Colors for different backend devices in the flowgraph visualization
// Each device has normal and selected states

/// @brief Color constants for CPU backend nodes
struct CPUColors {
    static constexpr U32 normal   = IM_COL32(224, 146,   0, 255);
    static constexpr U32 selected = IM_COL32(184, 119,   0, 255);
};

/// @brief Color constants for CUDA backend nodes
struct CUDAColors {
    static constexpr U32 normal   = IM_COL32( 95, 161,   2, 255);
    static constexpr U32 selected = IM_COL32( 85, 140,   2, 255);
};

/// @brief Color constants for Metal backend nodes
struct MetalColors {
    static constexpr U32 normal   = IM_COL32( 98,  60, 234, 255);
    static constexpr U32 selected = IM_COL32( 76,  33, 232, 255);
};

/// @brief Color constants for Vulkan backend nodes
struct VulkanColors {
    static constexpr U32 normal   = IM_COL32(238,  27,  52, 255);
    static constexpr U32 selected = IM_COL32(209,  16,  38, 255);
};

/// @brief Color constants for WebGPU backend nodes
struct WebGPUColors {
    static constexpr U32 normal   = IM_COL32( 59, 165, 147, 255);
    static constexpr U32 selected = IM_COL32( 49, 135, 121, 255);
};

/// @brief Color constants for disabled nodes
struct DisabledColors {
    static constexpr U32 normal   = IM_COL32( 75,  75,  75, 255);
    static constexpr U32 selected = IM_COL32( 75,  75,  75, 255);
};

/// @brief Default color constant
static constexpr U32 DefaultColor = IM_COL32(255, 255, 255, 255);

// =============================================================================
// Font Management
// =============================================================================

/// @brief Font resources for the compositor UI
/// Includes body, heading, and bold fonts for different UI contexts
struct Fonts {
    ImFont* body = nullptr;
    ImFont* h1 = nullptr;
    ImFont* h2 = nullptr;
    ImFont* bold = nullptr;
};

/// @brief Markdown configuration with font and styling settings
struct MarkdownContext {
    ImGui::MarkdownConfig config;
    Fonts fonts;
    F32 previousScalingFactor = 1.0f;
};

// =============================================================================
// Styling Functions
// =============================================================================
// These functions are implemented in src/compositor/imgui.cc, imnodes.cc, and markdown.cc

/// @brief Load fonts for the compositor UI
/// @param fonts Font structure to populate
void LoadFonts(Fonts& fonts);

/// @brief Setup ImGui styling
void ImGuiSetup();

/// @brief Scale ImGui styling based on DPI
/// @param scalingFactor Display scaling factor
void ImGuiScale(F32 scalingFactor);

/// @brief Setup ImNodes styling
void ImNodesSetup();

/// @brief Scale ImNodes styling based on DPI
/// @param scalingFactor Display scaling factor
void ImNodesScale(F32 scalingFactor);

/// @brief Setup ImGuiMarkdown styling
/// @param context Markdown context with configuration
void ImGuiMarkdownSetup(MarkdownContext& context);

/// @brief Markdown link callback
/// @param data Link callback data from ImGui::Markdown
void ImGuiMarkdownLinkCallback(ImGui::MarkdownLinkCallbackData data);

/// @brief Markdown format callback
/// @param md_info Format information from ImGui::Markdown
/// @param start Whether formatting is starting or ending
void ImGuiMarkdownFormatCallback(const ImGui::MarkdownFormatInfo& md_info, bool start);

}  // namespace Jetstream::CompositorStyling

#endif
