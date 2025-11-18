#ifndef JETSTREAM_COMPOSITOR_STATE_HH
#define JETSTREAM_COMPOSITOR_STATE_HH

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "jetstream/compositor/types.hh"
#include "jetstream/render/base.hh"
#include "jetstream/types.hh"

namespace Jetstream {

// Forward declarations
namespace Render {
class Texture;
}

namespace CompositorState {

/// @brief Runtime state for the compositor UI
/// This structure contains all mutable state used during compositor operation,
/// separated from the main Compositor class to improve organization and clarity.
struct Runtime {
    // === UI State Flags ===
    bool running = true;
    bool graphSpatiallyOrganized = false;
    bool rightClickMenuEnabled = false;
    bool sourceEditorEnabled = false;
    bool moduleStoreEnabled = true;
    bool infoPanelEnabled = true;
    bool flowgraphEnabled = true;
    bool debugDemoEnabled = false;
    bool debugLatencyEnabled = false;
    bool debugViewportEnabled = false;
    bool fullscreenEnabled = false;
    bool debugEnableTrace = false;
    bool benchmarkRunning = false;
    bool flowgraphSourceDirty = false;
    bool globalModalToggle = false;

    // === Node Interaction State ===
    I32 nodeDragId = -1;
    I32 nodeContextMenuNodeId = -1;
    U64 globalModalContentId = 0;
    ImGuiID mainNodeId = 0;

    // === Flowgraph File State ===
    std::string filenameField;
    std::string flowgraphFilename;
    std::vector<char> flowgraphBlob;
    std::string flowgraphSourceBuffer;

    // === Block Rename State ===
    Locale renameBlockLocale;
    std::string renameBlockNewId;

    // === Synchronization ===
    std::atomic_flag interfaceHalt{false};

    // === Image Assets ===
    std::shared_ptr<Render::Texture> primaryBannerTexture;
    std::shared_ptr<Render::Texture> secondaryBannerTexture;
};

/// @brief Data structures for compositor graph management
/// Caches and mappings for efficient node/link/pin lookups
struct GraphData {
    using LinkId = CompositorDetail::LinkId;
    using PinId = CompositorDetail::PinId;
    using NodeId = CompositorDetail::NodeId;
    using NodeState = CompositorDetail::NodeState;

    // Node state and topology
    std::unordered_map<Locale, NodeState, Locale::Hasher> nodeStates;
    std::unordered_map<Locale, std::vector<Locale>, Locale::Hasher> outputInputCache;
    std::vector<std::vector<std::vector<NodeId>>> nodeTopology;
    std::unordered_map<std::string, std::pair<bool, ImGuiID>> stacks;

    // Bidirectional mappings for efficient lookups
    std::unordered_map<LinkId, std::pair<Locale, Locale>> linkLocaleMap;
    std::unordered_map<Locale, PinId, Locale::Hasher> inputLocalePinMap;
    std::unordered_map<Locale, PinId, Locale::Hasher> outputLocalePinMap;
    std::unordered_map<PinId, Locale> pinLocaleMap;
    std::unordered_map<NodeId, Locale> nodeLocaleMap;
};

/// @brief Mailbox system for compositor actions
/// Uses optional<T> pattern for asynchronous message passing between UI and backend
struct Mailboxes {
    using CreateBlockMail = CompositorDetail::CreateBlockMail;
    using LinkMail = CompositorDetail::LinkMail;
    using UnlinkMail = CompositorDetail::UnlinkMail;
    using DeleteBlockMail = CompositorDetail::DeleteBlockMail;
    using ReloadBlockMail = CompositorDetail::ReloadBlockMail;
    using RenameBlockMail = CompositorDetail::RenameBlockMail;
    using ChangeBlockBackendMail = CompositorDetail::ChangeBlockBackendMail;
    using ChangeBlockDataTypeMail = CompositorDetail::ChangeBlockDataTypeMail;
    using ToggleBlockMail = CompositorDetail::ToggleBlockMail;

    // Block creation staging area
    CreateBlockMail createBlockStagingMailbox;

    // Action mailboxes (optional pattern for async communication)
    std::optional<LinkMail> linkMailbox;
    std::optional<UnlinkMail> unlinkMailbox;
    std::optional<CreateBlockMail> createBlockMailbox;
    std::optional<DeleteBlockMail> deleteBlockMailbox;
    std::optional<ReloadBlockMail> reloadBlockMailbox;
    std::optional<RenameBlockMail> renameBlockMailbox;
    std::optional<ChangeBlockBackendMail> changeBlockBackendMailbox;
    std::optional<ChangeBlockDataTypeMail> changeBlockDataTypeMailbox;
    std::optional<ToggleBlockMail> toggleBlockMailbox;

    // Flowgraph operation mailboxes
    std::optional<bool> resetFlowgraphMailbox;
    std::optional<bool> closeFlowgraphMailbox;
    std::optional<bool> openFlowgraphPathMailbox;
    std::optional<const char*> openFlowgraphBlobMailbox;
    std::optional<bool> updateFlowgraphBlobMailbox;
    std::optional<bool> saveFlowgraphMailbox;
    std::optional<bool> newFlowgraphMailbox;
    std::optional<bool> exitFullscreenMailbox;
    std::optional<std::vector<char>> applyFlowgraphMailbox;
};

}  // namespace CompositorState
}  // namespace Jetstream

#endif
