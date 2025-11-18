#ifndef JETSTREAM_COMPOSITOR_HH
#define JETSTREAM_COMPOSITOR_HH

#include "jetstream/types.hh"
#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/compositor/types.hh"
#include "jetstream/compositor/state.hh"
#include "jetstream/compositor/styling.hh"

// Compositor Architecture:
//
// The compositor manages the UI layer for the CyberEther flowgraph editor.
// It has been split into multiple focused headers for better maintainability:
//
// - compositor.hh (this file)  : Main interface and coordination
// - compositor/types.hh        : Core type definitions and mailbox structures
// - compositor/state.hh        : Runtime state, flags, and data structures
// - compositor/styling.hh      : UI colors, fonts, and styling functions
//
// Implementation files:
// - src/compositor/base.cc     : Main logic (3276 lines)
// - src/compositor/imgui.cc    : ImGui styling implementation
// - src/compositor/imnodes.cc  : ImNodes styling implementation
// - src/compositor/markdown.cc : Markdown styling implementation

namespace Jetstream {

class Instance;

class JETSTREAM_API Compositor {
 public:
    Compositor(Instance& instance);
    ~Compositor();

    // === Fluent Configuration API ===

    Compositor& showStore(const bool& enabled) {
        state.moduleStoreEnabled = enabled;
        return *this;
    }

    Compositor& showFlowgraph(const bool& enabled) {
        state.flowgraphEnabled = enabled;
        return *this;
    }

    // === Block Management ===

    Result addBlock(const Locale& locale,
                    const std::shared_ptr<Block>& block,
                    const Parser::RecordMap& inputMap,
                    const Parser::RecordMap& outputMap,
                    const Parser::RecordMap& stateMap,
                    const Block::Fingerprint& fingerprint);
    Result removeBlock(const Locale& locale);
    Result destroy();

    // === Rendering and Interaction ===

    Result draw();
    Result processInteractions();

    // === Accessors ===

    constexpr ImGui::MarkdownConfig& markdownConfig() {
        return styling.config;
    }

 private:
    // === Private Methods ===

    void lock();
    void unlock();

    Result refreshState();
    Result updateAutoLayoutState();
    Result checkAutoLayoutState();

    Result drawStatic();
    Result drawFlowgraph();

    Result loadImageAsset(const uint8_t* binary_data,
                          const U64& binary_len,
                          std::shared_ptr<Render::Texture>& texture);

    void ImGuiLoadFonts();
    void ImGuiStyleSetup();
    void ImGuiStyleScale();
    void ImNodesStyleSetup();
    void ImNodesStyleScale();
    void ImGuiMarkdownStyleSetup();

    static void ImGuiMarkdownLinkCallback(ImGui::MarkdownLinkCallbackData data);
    static void ImGuiMarkdownFormatCallback(const ImGui::MarkdownFormatInfo& md_info, bool start);

    // === Member Variables ===

    Instance& instance;

    // Organized state using helper structures
    CompositorState::Runtime state;
    CompositorState::GraphData graph;
    CompositorState::Mailboxes mailboxes;
    CompositorStyling::MarkdownContext styling;
};

}  // namespace Jetstream

#endif
