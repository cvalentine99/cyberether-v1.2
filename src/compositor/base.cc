#include <regex>

#include "jetstream/benchmark.hh"
#include "jetstream/compositor.hh"
#include "jetstream/instance.hh"
#include "jetstream/store.hh"
#include "jetstream/platform.hh"
#include "jetstream/render/tools/imnodes.h"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <stb_image.h>
#include "resources/images/base.hh"

// Looks like GCC-13 has a false-positive bug that is quite annoying.
// Silencing this for now. This should be fixed in GCC-14.
#if defined(__GNUC__) && (__GNUC__ >= 13)
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

namespace Jetstream {

using NodeId = CompositorDetail::NodeId;
using LinkId = CompositorDetail::LinkId;
using PinId = CompositorDetail::PinId;

// Color constants from CompositorStyling
constexpr auto DefaultColor = CompositorStyling::DefaultColor;
constexpr auto DisabledColor = CompositorStyling::DisabledColors::normal;
constexpr auto DisabledColorSelected = CompositorStyling::DisabledColors::selected;
constexpr auto CPUColor = CompositorStyling::CPUColors::normal;
constexpr auto CPUColorSelected = CompositorStyling::CPUColors::selected;
constexpr auto CUDAColor = CompositorStyling::CUDAColors::normal;
constexpr auto CUDAColorSelected = CompositorStyling::CUDAColors::selected;
constexpr auto MetalColor = CompositorStyling::MetalColors::normal;
constexpr auto MetalColorSelected = CompositorStyling::MetalColors::selected;
constexpr auto VulkanColor = CompositorStyling::VulkanColors::normal;
constexpr auto VulkanColorSelected = CompositorStyling::VulkanColors::selected;
constexpr auto WebGPUColor = CompositorStyling::WebGPUColors::normal;
constexpr auto WebGPUColorSelected = CompositorStyling::WebGPUColors::selected;

Compositor::Compositor(Instance& instance)
     : instance(instance) {
    JST_DEBUG("[COMPOSITOR] Creating compositor.");

    // Load ImGui fonts.

    ImGuiLoadFonts();

    // Configure ImGui.

    ImGuiStyleSetup();
    ImGuiStyleScale();

    // Create ImNodes.

    ImNodes::CreateContext();
    ImNodesStyleSetup();
    ImNodesStyleScale();

    // Configure ImGuiMarkdown.

    ImGuiMarkdownStyleSetup();

    // Set default graph.stacks.
    graph.stacks["Graph"] = {true, 0};

    // Initialize state.
    JST_CHECK_THROW(refreshState());

    // Load assets.

    JST_CHECK_THROW(loadImageAsset(Resources::compositor_banner_primary_bin,
                                  Resources::compositor_banner_primary_len,
                                  state.primaryBannerTexture));
    JST_CHECK_THROW(loadImageAsset(Resources::compositor_banner_secondary_bin,
                                   Resources::compositor_banner_secondary_len,
                                   state.secondaryBannerTexture));
}

Compositor::~Compositor() {
    // Destroy assets.

    state.primaryBannerTexture->destroy();
    state.secondaryBannerTexture->destroy();

    // Destroy ImNodes.

    ImNodes::DestroyContext();
}

Result Compositor::addBlock(const Locale& locale,
                            const std::shared_ptr<Block>& block,
                            const Parser::RecordMap& inputMap,
                            const Parser::RecordMap& outputMap,
                            const Parser::RecordMap& stateMap,
                            const Block::Fingerprint& fingerprint) {
    JST_DEBUG("[COMPOSITOR] Adding block '{}'.", locale);

    // Make sure compositor is state.running.
    state.running = true;
    state.globalModalToggle = false;

    // Check if the locale is already created.
    if (graph.nodeStates.contains(locale)) {
        JST_ERROR("[COMPOSITOR] Entry for node '{}' already exists.", locale);
        return Result::ERROR;
    }

    // Prevent drawMainInterface from state.running.
    lock();

    // Save block in node state.

    auto& nodeState = graph.nodeStates[locale];

    nodeState.block = block;
    nodeState.inputMap = inputMap;
    nodeState.outputMap = outputMap;
    nodeState.stateMap = stateMap;
    nodeState.fingerprint = fingerprint;
    nodeState.title = jst::fmt::format("{} ({})", block->name(), locale);

    JST_CHECK(refreshState());

    // Resume drawMainInterface.
    unlock();

    return Result::SUCCESS;
}

Result Compositor::removeBlock(const Locale& locale) {
    JST_DEBUG("[COMPOSITOR] Removing block '{}'.", locale);

    // Return early if the scheduler is not state.running.
    if (!state.running) {
        return Result::SUCCESS;
    }

    // Check if the locale is already created.
    if (!graph.nodeStates.contains(locale)) {
        JST_ERROR("[COMPOSITOR] Entry for node '{}' doesn't exist.", locale);
        return Result::ERROR;
    }

    // Prevent drawMainInterface from state.running.
    lock();

    // Save block in node state.
    graph.nodeStates.erase(locale);

    JST_CHECK(refreshState());

    // Resume drawMainInterface.
    unlock();

    return Result::SUCCESS;
}

Result Compositor::refreshState() {
    JST_DEBUG("[COMPOSITOR] Refreshing interface state.");

    // Create interface state, input, output and pin cache.
    graph.pinLocaleMap.clear();
    graph.nodeLocaleMap.clear();
    graph.inputLocalePinMap.clear();
    graph.outputLocalePinMap.clear();
    graph.outputInputCache.clear();

    U64 id = 1;
    for (auto& [locale, state] : graph.nodeStates) {
        // Generate id for node.
        state.id = id;
        graph.nodeLocaleMap[id++] = locale;

        // Cleanup and create pin map and convert locale to interface locale.

        state.inputs.clear();
        state.outputs.clear();

        for (const auto& [inputPinId, inputRecord] : state.inputMap) {
            // Generate clean locale.
            const Locale& inputLocale = {locale.blockId, "", inputPinId};

            // Save the input pin locale.
            graph.pinLocaleMap[id] = inputLocale;
            graph.inputLocalePinMap[inputLocale] = id;

            // Generate clean locale.
            const Locale& outputLocale = inputRecord.locale.pin();

            // Save the incoming input locale.
            state.inputs[id++] = outputLocale;

            // Save the output to input locale map cache.
            graph.outputInputCache[outputLocale].push_back(inputLocale);
        }

        for (const auto& [outputPinId, outputRecord] : state.outputMap) {
            // Generate clean locale.
            const Locale& outputLocale = {locale.blockId, "", outputPinId};

            // Save the output pin locale.
            graph.pinLocaleMap[id] = outputLocale;
            graph.outputLocalePinMap[outputLocale] = id;

            // Save the outgoing output locale.
            state.outputs[id++] = outputLocale;
        }
    }

    // Create link and edges.
    graph.linkLocaleMap.clear();

    U64 linkId = 0;
    for (auto& [locale, state] : graph.nodeStates) {
        // Cleanup buffers.
        state.edges.clear();

        for (const auto& [_, inputLocale] : state.inputs) {
            if (graph.nodeStates.contains(inputLocale.block())) {
                state.edges.insert(graph.nodeStates.at(inputLocale.block()).id);
            }
        }

        for (const auto& [_, outputLocale] : state.outputs) {
            for (const auto& inputLocale : graph.outputInputCache[outputLocale]) {
                state.edges.insert(graph.nodeStates.at(inputLocale.block()).id);

                graph.linkLocaleMap[linkId++] = {inputLocale, outputLocale};
            }
        }
    }

    return Result::SUCCESS;
}

Result Compositor::checkAutoLayoutState() {
    JST_DEBUG("[COMPOSITOR] Checking auto layout state.");

    bool graphHasPos = false;
    for (const auto& [_, state] : graph.nodeStates) {
        if (state.block->getState().nodePos != Extent2D<F32>{0.0f, 0.0f}) {
            graphHasPos = true;
        }
    }

    if (!graphHasPos) {
        JST_CHECK(updateAutoLayoutState());
    }

    return Result::SUCCESS;
}

Result Compositor::updateAutoLayoutState() {
    JST_DEBUG("[COMPOSITOR] Updating auto layout state.");

    state.graphSpatiallyOrganized = false;

    // Separate graph in sub-graphs if applicable.
    U64 clusterCount = 0;
    std::unordered_set<NodeId> visited;

    for (const auto& [nodeId, _] : graph.nodeLocaleMap) {
        if (visited.contains(nodeId)) {
            continue;
        }

        std::stack<U64> stack;
        stack.push(nodeId);

        while (!stack.empty()) {
            U64 current = stack.top();
            stack.pop();

            if (visited.contains(current)) {
                continue;
            }

            visited.insert(current);
            auto& state = graph.nodeStates.at(graph.nodeLocaleMap.at(current));
            state.clusterLevel = clusterCount;
            for (const auto& neighbor : state.edges) {
                if (!visited.contains(neighbor)) {
                    stack.push(neighbor);
                }
            }
        }
        clusterCount += 1;
    }

    // Create automatic graph layout.
    graph.nodeTopology.clear();

    U64 columnId = 0;
    std::unordered_set<NodeId> S;

    for (const auto& [nodeId, nodeLocale] : graph.nodeLocaleMap) {
        if (graph.nodeStates.at(nodeLocale).inputs.size() == 0) {
            S.insert(nodeId);
        }
    }

    while (!S.empty()) {
        std::unordered_set<U64> nodesToInsert;
        std::unordered_set<U64> nodesToExclude;
        std::unordered_map<NodeId, std::unordered_set<NodeId>> nodeMatches;

        // Build the matches map.
        for (const auto& sourceNodeId : S) {
            const auto& outputList = graph.nodeStates.at(graph.nodeLocaleMap.at(sourceNodeId)).outputs;

            if (outputList.empty()) {
                nodesToInsert.insert(sourceNodeId);
                continue;
            }

            for (const auto& [_, outputLocale] : outputList) {
                const auto& inputList = graph.outputInputCache.at(outputLocale);

                if (inputList.empty()) {
                    nodesToInsert.insert(sourceNodeId);
                    continue;
                }

                for (const auto& inputLocale : inputList) {
                    nodeMatches[graph.nodeStates.at(inputLocale.block()).id].insert(sourceNodeId);
                }
            }
        }

        U64 previousSetSize = S.size();

        // Determine which nodes to insert and which to exclude.
        for (const auto& [targetNodeId, sourceNodes] : nodeMatches) {
            U64 inputCount = 0;
            for (const auto& [_, locale] : graph.nodeStates.at(graph.nodeLocaleMap.at(targetNodeId)).inputs) {
                if (!locale.empty()) {
                    inputCount += 1;
                }
            }
            if (inputCount >= sourceNodes.size()) {
                S.insert(targetNodeId);
                nodesToInsert.insert(sourceNodes.begin(), sourceNodes.end());
            } else {
                nodesToExclude.insert(sourceNodes.begin(), sourceNodes.end());
            }
        }

        // Exclude nodes from the nodesToInsert set.
        for (const auto& node : nodesToExclude) {
            nodesToInsert.erase(node);
        }

        // If no new nodes were added to S, insert all nodes from S into nodesToInsert.
        if (previousSetSize == S.size()) {
            nodesToInsert.insert(S.begin(), S.end());
        }

        for (const auto& nodeId : nodesToInsert) {
            const U64& clusterId = graph.nodeStates.at(graph.nodeLocaleMap[nodeId]).clusterLevel;
            if (graph.nodeTopology.size() <= clusterId) {
                graph.nodeTopology.resize(clusterId + 1);
            }
            if (graph.nodeTopology.at(clusterId).size() <= columnId) {
                graph.nodeTopology[clusterId].resize(columnId + 1);
            }
            graph.nodeTopology[clusterId][columnId].push_back(nodeId);
            S.erase(nodeId);
        }

        columnId += 1;
    }

    return Result::SUCCESS;
}

Result Compositor::destroy() {
    JST_DEBUG("[COMPOSITOR] Destroying compositor.");

    // Stop execution.
    state.running = false;

    // Acquire lock.
    lock();

    // Reseting variables.

    state.rightClickMenuEnabled = false;
    state.graphSpatiallyOrganized = false;
    state.sourceEditorEnabled = false;
    state.moduleStoreEnabled = true;
    state.debugDemoEnabled = false;
    state.debugLatencyEnabled = false;
    state.debugViewportEnabled = false;
    state.flowgraphEnabled = true;
    state.infoPanelEnabled = true;
    state.globalModalContentId = 0;
    state.nodeContextMenuNodeId = 0;
    state.flowgraphFilename = {};
    state.flowgraphBlob = {};

    // Clearing buffers.

    graph.linkLocaleMap.clear();
    graph.inputLocalePinMap.clear();
    graph.outputLocalePinMap.clear();
    graph.pinLocaleMap.clear();
    graph.nodeLocaleMap.clear();
    graph.stacks.clear();
    graph.nodeTopology.clear();
    graph.outputInputCache.clear();
    graph.nodeStates.clear();

    // Add static.
    graph.stacks["Graph"] = {true, 0};

    // Release lock.
    unlock();

    // Reseting locks.

    state.interfaceHalt.clear();
    state.interfaceHalt.notify_all();

    return Result::SUCCESS;
}

Result Compositor::loadImageAsset(const uint8_t* binary_data,
                                  const U64& binary_len,
                                  std::shared_ptr<Render::Texture>& texture) {
    int image_width, image_height, image_channels;

    uint8_t* raw_data = stbi_load_from_memory(
        binary_data,
        binary_len,
        &image_width,
        &image_height,
        &image_channels,
        4
    );
    if (raw_data == nullptr) {
        JST_FATAL("[COMPOSITOR] Could not load image asset.");
        return Result::ERROR;
    }

    Render::Texture::Config config;
    config.size = {
        static_cast<U64>(image_width),
        static_cast<U64>(image_height)
    };
    config.buffer = static_cast<uint8_t*>(raw_data);
    JST_CHECK(instance.window().build(texture, config));
    JST_CHECK(texture->create());
    stbi_image_free(raw_data);

    return Result::SUCCESS;
}

Result Compositor::draw() {
    // Prevent state from refreshing while drawing these methods.

    state.interfaceHalt.wait(true);
    state.interfaceHalt.test_and_set();

    // Scale ImGui and ImNodes styles.

    if (instance.window().scalingFactor() != styling.previousScalingFactor) {
        ImGuiStyleScale();
        ImNodesStyleScale();
        styling.previousScalingFactor = instance.window().scalingFactor();
    }

    // Draw main interface.

    JST_CHECK(drawStatic());
    JST_CHECK(drawFlowgraph());

    // Release lock.

    state.interfaceHalt.clear();
    state.interfaceHalt.notify_one();

    return Result::SUCCESS;
}

Result Compositor::processInteractions() {
    if (mailboxes.createBlockMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, metadata = *mailboxes.createBlockMailbox](){
//             ImGui::InsertNotification({ ImGuiToastType_Info, 2500, "Adding module..." });  // TODO: Re-enable when imgui-notify is available

            // Create new node fingerprint.
            const auto& [module, device] = metadata;
            const auto moduleEntry = Store::BlockMetadataList().at(module);
            if (moduleEntry.options.at(device).empty()) {
                //  ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "No compatible data types for this module." });  // TODO: Re-enable when imgui-notify is available
                return; // No compatible data types
            }
            const auto& [inputDataType, outputDataType] = moduleEntry.options.at(device).at(0);

            Block::Fingerprint fingerprint = {};
            fingerprint.id = module;
            fingerprint.device = GetDeviceName(device);
            fingerprint.inputDataType = inputDataType;
            fingerprint.outputDataType = outputDataType;

            // Create node where the mouse dropped the module.
            Parser::RecordMap configMap, inputMap, stateMap;
            const auto [x, y] = ImNodes::ScreenSpaceToGridSpace(ImGui::GetMousePos());
            const auto& scalingFactor = instance.window().scalingFactor();
            stateMap["nodePos"] = {Extent2D<F32>{x / scalingFactor, y / scalingFactor}};

            // Create module.
            JST_CHECK_NOTIFY(Store::BlockConstructorList().at(fingerprint)(instance, "", configMap, inputMap, stateMap));

            // Update source.
            mailboxes.updateFlowgraphBlobMailbox = true;
        });

        mailboxes.createBlockMailbox.reset();
    }

    if (mailboxes.deleteBlockMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, block = *mailboxes.deleteBlockMailbox](){
            JST_CHECK_NOTIFY(instance.removeBlock(block));
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.deleteBlockMailbox.reset();
    }

    if (mailboxes.renameBlockMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, metadata = *mailboxes.renameBlockMailbox](){
            const auto& [locale, id] = metadata;
//             ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Renaming block..." });  // TODO: Re-enable when imgui-notify is available
            JST_CHECK_NOTIFY(instance.renameBlock(locale, id));
        });
        mailboxes.renameBlockMailbox.reset();
    }

    if (mailboxes.reloadBlockMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, locale = *mailboxes.reloadBlockMailbox](){
//             ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reloading block..." });  // TODO: Re-enable when imgui-notify is available
            JST_CHECK_NOTIFY(instance.reloadBlock(locale));
        });
        mailboxes.reloadBlockMailbox.reset();
    }

    if (mailboxes.linkMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, metadata = *mailboxes.linkMailbox](){
            const auto& [inputLocale, outputLocale] = metadata;
            JST_CHECK_NOTIFY(instance.linkBlocks(inputLocale, outputLocale));
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.linkMailbox.reset();
    }

    if (mailboxes.unlinkMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, metadata = *mailboxes.unlinkMailbox](){
            const auto& [inputLocale, outputLocale] = metadata;
            JST_CHECK_NOTIFY(instance.unlinkBlocks(inputLocale, outputLocale));
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.unlinkMailbox.reset();
    }

    if (mailboxes.changeBlockBackendMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, metadata = *mailboxes.changeBlockBackendMailbox](){
            const auto& [locale, device] = metadata;
            JST_CHECK_NOTIFY(instance.changeBlockBackend(locale, device));
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.changeBlockBackendMailbox.reset();
    }

    if (mailboxes.changeBlockDataTypeMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, metadata = *mailboxes.changeBlockDataTypeMailbox](){
            const auto& [locale, type] = metadata;
            JST_CHECK_NOTIFY(instance.changeBlockDataType(locale, type));
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.changeBlockDataTypeMailbox.reset();
    }

    if (mailboxes.toggleBlockMailbox.has_value()) {
        const auto& [locale, enable] = *mailboxes.toggleBlockMailbox;
        if (graph.nodeStates.contains(locale.block())) {
            auto& state = graph.nodeStates.at(locale.block());
            if (!enable && state.enabled) {
                state.cachedState = state.block->state;
                state.hasCachedState = true;
                state.block->state.viewEnabled = false;
                state.block->state.controlEnabled = false;
                state.block->state.previewEnabled = false;
                state.block->state.fullscreenEnabled = false;
                state.enabled = false;
            } else if (enable && !state.enabled) {
                state.enabled = true;
                if (state.hasCachedState) {
                    state.block->state = state.cachedState;
                } else {
                    state.block->state.viewEnabled = state.block->shouldDrawView();
                    state.block->state.controlEnabled = state.block->shouldDrawControl();
                    state.block->state.previewEnabled = state.block->shouldDrawPreview();
                    state.block->state.fullscreenEnabled = false;
                }
            } else {
                state.enabled = enable;
            }

            const char* toast = enable ? "Node enabled." : "Node disabled.";
//             ImGui::InsertNotification({ enable ? ImGuiToastType_Success  // TODO: Re-enable when imgui-notify is available
//                                                : ImGuiToastType_Info,
//                                         2000,
//                                         toast });
            mailboxes.updateFlowgraphBlobMailbox = true;
        }
        mailboxes.toggleBlockMailbox.reset();
    }

    if (mailboxes.resetFlowgraphMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&](){
            JST_CHECK_NOTIFY(instance.reset());
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.resetFlowgraphMailbox.reset();
    }

    if (mailboxes.closeFlowgraphMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&](){
            JST_CHECK_NOTIFY([&]{
                JST_CHECK(instance.reset());
                JST_CHECK(instance.flowgraph().destroy());
                return Result::SUCCESS;
            }());
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.closeFlowgraphMailbox.reset();
    }

    if (mailboxes.openFlowgraphPathMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&](){
//             ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Loading flowgraph..." });  // TODO: Re-enable when imgui-notify is available
            JST_CHECK_NOTIFY([&]{
                JST_CHECK(instance.flowgraph().create());
                JST_CHECK(instance.flowgraph().importFromFile(state.flowgraphFilename));
                JST_CHECK(checkAutoLayoutState());
                return Result::SUCCESS;
            }());
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.openFlowgraphPathMailbox.reset();
    }

    if (mailboxes.openFlowgraphBlobMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&, string = *mailboxes.openFlowgraphBlobMailbox](){
//             ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Loading flowgraph..." });  // TODO: Re-enable when imgui-notify is available
            JST_CHECK_NOTIFY([&]{
                JST_CHECK(instance.flowgraph().create());
                std::vector<char> blob(string, string + strlen(string));
                JST_CHECK(instance.flowgraph().importFromBlob(blob));
                JST_CHECK(checkAutoLayoutState());
                return Result::SUCCESS;
            }());
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.openFlowgraphBlobMailbox.reset();
    }

    if (mailboxes.saveFlowgraphMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&](){
//             ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Saving flowgraph..." });  // TODO: Re-enable when imgui-notify is available
            JST_CHECK_NOTIFY(instance.flowgraph().exportToFile(state.flowgraphFilename));
        });
        mailboxes.saveFlowgraphMailbox.reset();
    }

    if (mailboxes.newFlowgraphMailbox.has_value()) {
        JST_DISPATCH_ASYNC([&](){
            JST_CHECK_NOTIFY(instance.flowgraph().create());
            mailboxes.updateFlowgraphBlobMailbox = true;
        });
        mailboxes.newFlowgraphMailbox.reset();
    }

    if (mailboxes.updateFlowgraphBlobMailbox.has_value()) {
        JST_CHECK(instance.flowgraph().exportToBlob(state.flowgraphBlob));
        if (state.sourceEditorEnabled) {
            state.flowgraphSourceBuffer.assign(state.flowgraphBlob.begin(), state.flowgraphBlob.end());
            state.flowgraphSourceDirty = false;
        }
        mailboxes.updateFlowgraphBlobMailbox.reset();
    }

    if (mailboxes.applyFlowgraphMailbox.has_value()) {
        auto blob = std::move(*mailboxes.applyFlowgraphMailbox);
        mailboxes.applyFlowgraphMailbox.reset();
        JST_CHECK(instance.flowgraph().create());
        JST_CHECK(instance.flowgraph().importFromBlob(blob));
        JST_CHECK(checkAutoLayoutState());
        state.flowgraphBlob = blob;
        state.flowgraphSourceBuffer.assign(state.flowgraphBlob.begin(), state.flowgraphBlob.end());
        state.flowgraphSourceDirty = false;
//         ImGui::InsertNotification({ImGuiToastType_Success, 2000, "Flowgraph source applied."});  // TODO: Re-enable when imgui-notify is available
    }

    return Result::SUCCESS;
}

Result Compositor::drawStatic() {
    // Load local variables.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const auto& io = ImGui::GetIO();
    const auto& scalingFactor = instance.window().scalingFactor();
    const bool flowgraphLoaded = instance.flowgraph().created();
    I32 interactionTrigger = 0;

    //
    // Grab Shortcuts.
    //

    const auto flag = ImGuiInputFlags_RouteAlways;

    if (ImGui::Shortcut(ImGuiKey_Escape, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Escape shortcut pressed.");
        interactionTrigger = 99;
    }

    if (ImGui::Shortcut(ImGuiMod_Super | ImGuiKey_S, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Save document shortcut pressed.");
        interactionTrigger = 1;
    }

    if (ImGui::Shortcut(ImGuiMod_Super | ImGuiKey_N, 0, flag)) {
        JST_TRACE("[COMPOSITOR] New document shortcut pressed.");
        interactionTrigger = 2;
    }

    if (ImGui::Shortcut(ImGuiMod_Super | ImGuiKey_O, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Open document shortcut pressed.");
        interactionTrigger = 3;
    }

    if (ImGui::Shortcut(ImGuiMod_Super | ImGuiKey_I, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Info document shortcut pressed.");
        interactionTrigger = 4;
    }

    if (ImGui::Shortcut(ImGuiMod_Super | ImGuiKey_W, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Close document shortcut pressed.");
        interactionTrigger = 5;
    }

    if (ImGui::Shortcut(ImGuiMod_Super | ImGuiKey_L, 0, flag)) {
        JST_TRACE("[COMPOSITOR] Exit fullscreen shortcut pressed.");
        interactionTrigger = 9;
    }

    //
    // Menu Bar.
    //

    F32 currentHeight = 0.0f;

    if (ImGui::BeginMainMenuBar()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.65f, 0.75f, 0.62f, 1.0f));
        if (ImGui::BeginMenu("CyberEther")) {
            ImGui::PopStyleColor();

            if (ImGui::MenuItem("About CyberEther")) {
                state.globalModalToggle = true;
                state.globalModalContentId = 0;
            }
            if (ImGui::MenuItem("View License")) {
                state.globalModalToggle = true;
                state.globalModalContentId = 7;
            }
            if (ImGui::MenuItem("Third-Party OSS")) {
                state.globalModalToggle = true;
                state.globalModalContentId = 8;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Block Backend Matrix")) {
                state.globalModalToggle = true;
                state.globalModalContentId = 1;
            }
#ifndef JST_OS_BROWSER
            ImGui::Separator();
            if (ImGui::MenuItem("Quit CyberEther")) {
                std::exit(0);
            }
#endif
            ImGui::EndMenu();
        } else {
            ImGui::PopStyleColor();
        }

        if (ImGui::BeginMenu("Flowgraph")) {
            if (ImGui::MenuItem("New", "CTRL+N", false, !flowgraphLoaded)) {
                interactionTrigger = 2;
            }
            if (ImGui::MenuItem("Open", "CTRL+O", false, !flowgraphLoaded)) {
                interactionTrigger = 3;
            }
            if (ImGui::MenuItem("Save", "CTRL+S", false, flowgraphLoaded)) {
                interactionTrigger = 1;
            }
            if (ImGui::MenuItem("Info", "CTRL+I", false, flowgraphLoaded)) {
                interactionTrigger = 4;
            }
            if (ImGui::MenuItem("Close", "CTRL+W", false, flowgraphLoaded)) {
                interactionTrigger = 5;
            }
            if (ImGui::MenuItem("Rename", nullptr, false, flowgraphLoaded)) {
                interactionTrigger = 8;
            }
            if (ImGui::MenuItem("Reset", nullptr, false, flowgraphLoaded)) {
                interactionTrigger = 6;
            }
            if (ImGui::MenuItem("Source", nullptr, false, flowgraphLoaded)) {
                interactionTrigger = 7;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Show Info Panel", nullptr, &state.infoPanelEnabled)) { }
            if (ImGui::MenuItem("Show Flowgraph Source", nullptr, &state.sourceEditorEnabled, flowgraphLoaded)) { }
            if (ImGui::MenuItem("Show Flowgraph", nullptr, &state.flowgraphEnabled, flowgraphLoaded)) { }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Developer")) {
            if (ImGui::MenuItem("Show Demo Window", nullptr, &state.debugDemoEnabled)) { }
            if (ImGui::MenuItem("Show Latency Window", nullptr, &state.debugLatencyEnabled)) { }
            if (ImGui::MenuItem("Show Viewport Window", nullptr, &state.debugViewportEnabled)) { }
            if (ImGui::MenuItem("Enable Trace", nullptr, &state.debugEnableTrace)) {
                if (state.debugEnableTrace) {
                    JST_LOG_SET_DEBUG_LEVEL(4);
                } else {
                    JST_LOG_SET_DEBUG_LEVEL(JST_LOG_DEBUG_DEFAULT_LEVEL);
                }
            }
            if (ImGui::MenuItem("Open Benchmarking Tool", nullptr, nullptr)) {
                state.globalModalToggle = true;
                state.globalModalContentId = 9;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Getting started")) {
                state.globalModalToggle = true;
                state.globalModalContentId = 2;
            }
            if (ImGui::MenuItem("Luigi's Twitter")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://twitter.com/luigifcruz"));
            }
            if (ImGui::MenuItem("Documentation")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther"));
            }
            if (ImGui::MenuItem("Open repository")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther"));
            }
            if (ImGui::MenuItem("Report an issue")) {
                JST_CHECK_NOTIFY(Platform::OpenUrl("https://github.com/luigifcruz/CyberEther/issues"));
            }
            ImGui::Separator();
            if (ImGui::MenuItem("View license")) {
                state.globalModalToggle = true;
                state.globalModalContentId = 7;
            }
            if (ImGui::MenuItem("Third-Party OSS")) {
                state.globalModalToggle = true;
                state.globalModalContentId = 8;
            }
            ImGui::EndMenu();
        }

        currentHeight += ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    //
    // Tool Bar.
    //

    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, currentHeight));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, 0));

    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("##ToolBar", nullptr, ImGuiWindowFlags_NoDecoration |
                                           ImGuiWindowFlags_NoNav |
                                           ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoSavedSettings);
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();

        {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));

            if (!flowgraphLoaded) {
                if (ImGui::Button(ICON_FA_FILE " New")) {
                    interactionTrigger = 2;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open")) {
                    interactionTrigger = 3;
                }
                ImGui::SameLine();
            } else {
                if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save")) {
                    interactionTrigger = 1;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_CIRCLE_XMARK " Close")) {
                    interactionTrigger = 5;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_ERASER " Reset")) {
                    interactionTrigger = 6;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_FILE_CODE " Source")) {
                    interactionTrigger = 7;
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_CIRCLE_INFO " Info")) {
                    interactionTrigger = 4;
                }
                ImGui::SameLine();

                ImGui::Dummy(ImVec2(5.0f, 0.0f));
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_HAND_SPARKLES " Auto Layout")) {
                    JST_CHECK_NOTIFY(updateAutoLayoutState());
                }
                ImGui::SameLine();

                if (ImGui::Button(ICON_FA_LAYER_GROUP " New Stack")) {
                    graph.stacks[jst::fmt::format("Stack #{}", graph.stacks.size())] = {true, 0};
                }
                ImGui::SameLine();
            }

#ifdef JST_OS_BROWSER
            ImGui::Dummy(ImVec2(5.0f, 0.0f));
            ImGui::SameLine();

            if (ImGui::Button(ICON_FA_PLUG " Connect WebUSB Device")) {
                if (EM_ASM_INT({ return 'usb' in navigator; }) == 0) {
//                     ImGui::InsertNotification({ ImGuiToastType_Error, 10000, "This browser is not compatible with WebUSB. "  // TODO: Re-enable when imgui-notify is available
//                                                                              "Try a Chromium based browser like Chrome, Brave, or Opera GX." });
                } else {
                    EM_ASM({  openUsbDevice(); });
                }
            }
            ImGui::SameLine();
#endif

            if (flowgraphLoaded) {
                ImGui::Dummy(ImVec2(5.0f, 0.0f));
                ImGui::SameLine();

                const auto& title = instance.flowgraph().title();
                if (title.empty()) {
                    ImGui::TextUnformatted("Title: N/A");
                } else {
                    ImGui::TextFormatted("Title: {}", title);
                }

                ImGui::SameLine();
                ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);

                    std::string textSummary, textAuthor, textLicense, textDescription;

                    if (!instance.flowgraph().summary().empty()) {
                        textSummary = jst::fmt::format("Summary: {}\n", instance.flowgraph().summary());
                    }

                    if (!instance.flowgraph().author().empty()) {
                        textAuthor = jst::fmt::format("Author:  {}\n", instance.flowgraph().author());
                    }

                    if (!instance.flowgraph().license().empty()) {
                        textLicense = jst::fmt::format("License: {}\n", instance.flowgraph().license());
                    }

                    if (!instance.flowgraph().description().empty()) {
                        textDescription = jst::fmt::format("Description:\n{}", instance.flowgraph().description());
                    }

                    ImGui::TextWrapped("%s%s%s%s", textSummary.c_str(),
                                                   textAuthor.c_str(),
                                                   textLicense.c_str(),
                                                   textDescription.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }

                ImGui::SameLine();
            }

            // Spotlight-style module search in toolbar
            if (flowgraphLoaded) {
                static std::string moduleSearchText = "";
                static bool showModuleDropdown = false;
                static bool searchInputActive = false;
                static ImVec2 searchInputPos;
                static int selectedModuleIndex = -1;
                static std::vector<std::pair<std::string, Device>> filteredModules;
                static bool moduleAttachMode = false;
                static std::pair<std::string, Device> moduleToAttach;
                static bool shouldScrollToSelected = false;

                const F32 searchWidth = 350.0f * scalingFactor;

                const F32 totalWidth = searchWidth + 10.0f * scalingFactor;
                const F32 availableWidth = ImGui::GetWindowWidth() - ImGui::GetCursorPosX();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availableWidth - totalWidth);

                if (ImGui::IsKeyPressed(ImGuiKey_Slash, false)) {
                    ImGui::SetKeyboardFocusHere();
                    moduleSearchText.clear();
                    showModuleDropdown = true;
                    selectedModuleIndex = -1;

                    filteredModules.clear();
                    for (const auto& [id, module] : Store::BlockMetadataList("")) {
                        for (const auto& [device, _] : module.options) {
                            filteredModules.push_back({id, device});
                        }
                    }
                    interactionTrigger = false;
                }

                ImGui::SetNextItemWidth(searchWidth);
                searchInputPos = ImGui::GetCursorScreenPos();

                static char searchBuffer[256];
                if (moduleSearchText.size() < sizeof(searchBuffer)) {
                    strcpy(searchBuffer, moduleSearchText.c_str());
                }
                bool inputChanged = ImGui::InputTextWithHint("##ModuleSearch", ICON_FA_MAGNIFYING_GLASS " Search blocks... (/)", searchBuffer, sizeof(searchBuffer));
                if (inputChanged) {
                    moduleSearchText = searchBuffer;
                }
                bool inputActive = ImGui::IsItemActive();
                bool inputFocused = ImGui::IsItemFocused();

                if ((inputFocused || inputActive) && showModuleDropdown) {
                    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                        showModuleDropdown = false;
                        selectedModuleIndex = -1;
                        moduleSearchText.clear();
                        moduleAttachMode = false;
                    } else if (!filteredModules.empty()) {
                        bool navigationKeyPressed = false;
                        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
                            selectedModuleIndex = (selectedModuleIndex + 1) % (int)filteredModules.size();
                            navigationKeyPressed = true;
                        } else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
                            selectedModuleIndex = selectedModuleIndex <= 0 ? (int)filteredModules.size() - 1 : selectedModuleIndex - 1;
                            navigationKeyPressed = true;
                        } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) && selectedModuleIndex >= 0 && static_cast<size_t>(selectedModuleIndex) < filteredModules.size()) {
                            auto selectedPair = filteredModules[selectedModuleIndex];

                            moduleToAttach = selectedPair;
                            moduleAttachMode = true;
                            showModuleDropdown = false;
                            selectedModuleIndex = -1;
                            moduleSearchText.clear();
                        }

                        if (navigationKeyPressed) {
                            shouldScrollToSelected = true;
                        }
                    }
                }

                if (inputChanged || (inputActive && !searchInputActive)) {
                    showModuleDropdown = true;
                    selectedModuleIndex = 0;

                    filteredModules.clear();
                    const char* filterText = moduleSearchText.empty() ? "" : moduleSearchText.c_str();
                    for (const auto& [id, module] : Store::BlockMetadataList(filterText)) {
                        for (const auto& [device, _] : module.options) {
                            filteredModules.push_back({id, device});
                        }
                    }
                    if (filteredModules.empty()) {
                        selectedModuleIndex = -1;
                    }
                }
                searchInputActive = inputActive;

                if (moduleAttachMode) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

                    ImVec2 mousePos = ImGui::GetMousePos();
                    ImDrawList* drawList = ImGui::GetForegroundDrawList();

                    const auto& [moduleId, device] = moduleToAttach;
                    auto blockMetadataList = Store::BlockMetadataList("");
                    const auto moduleEntry = blockMetadataList.find(moduleId);
                    if (moduleEntry != blockMetadataList.end()) {
                        std::string displayText = ICON_FA_CUBE " " + moduleEntry->second.title + " (" + std::string(GetDevicePrettyName(device)) + ")";

                        ImVec2 textSize = ImGui::CalcTextSize(displayText.c_str());
                        ImVec2 tooltipSize = ImVec2(textSize.x + 16.0f * scalingFactor, textSize.y + 12.0f * scalingFactor);
                        ImVec2 tooltipPos = ImVec2(mousePos.x + 5.0f * scalingFactor, mousePos.y - tooltipSize.y - 5.0f * scalingFactor);

                        drawList->AddRectFilled(tooltipPos,
                                               ImVec2(tooltipPos.x + tooltipSize.x, tooltipPos.y + tooltipSize.y),
                                               IM_COL32(45, 45, 48, 240),
                                               4.0f * scalingFactor);

                        drawList->AddRect(tooltipPos,
                                         ImVec2(tooltipPos.x + tooltipSize.x, tooltipPos.y + tooltipSize.y),
                                         IM_COL32(80, 80, 80, 255),
                                         4.0f * scalingFactor,
                                         0,
                                         1.0f);

                        U32 deviceColor;
                        switch (device) {
                            case Device::CPU: deviceColor = CPUColor; break;
                            case Device::CUDA: deviceColor = CUDAColor; break;
                            case Device::Metal: deviceColor = MetalColor; break;
                            case Device::Vulkan: deviceColor = VulkanColor; break;
                            case Device::WebGPU: deviceColor = WebGPUColor; break;
                            default: deviceColor = IM_COL32(255, 255, 255, 255);
                        }

                        drawList->AddText(ImVec2(tooltipPos.x + 8.0f * scalingFactor, tooltipPos.y + 6.0f * scalingFactor),
                                         deviceColor,
                                         displayText.c_str());
                    }

                    if (ImGui::IsMouseClicked(0) || ImGui::IsMouseReleased(0)) {
                        mailboxes.createBlockStagingMailbox = moduleToAttach;
                        mailboxes.createBlockMailbox = moduleToAttach;
                        moduleAttachMode = false;
                    } else if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                        moduleAttachMode = false;
                    }
                }

                if (!moduleAttachMode && !inputActive && !inputFocused && ImGui::IsMouseClicked(0)) {
                    ImVec2 mousePos = ImGui::GetMousePos();
                    ImVec2 dropdownPos = ImVec2(searchInputPos.x, searchInputPos.y + ImGui::GetFrameHeight());
                    const F32 dropdownWidth = searchWidth;
                    const F32 dropdownHeight = 400.0f * scalingFactor;

                    if (mousePos.x < dropdownPos.x || mousePos.x > dropdownPos.x + dropdownWidth ||
                        mousePos.y < dropdownPos.y || mousePos.y > dropdownPos.y + dropdownHeight) {
                        showModuleDropdown = false;
                        selectedModuleIndex = -1;
                    }
                }

                if (showModuleDropdown) {
                    const F32 dropdownWidth = searchWidth;
                    const F32 dropdownHeight = 400.0f * scalingFactor;

                    ImVec2 dropdownPos = ImVec2(searchInputPos.x, searchInputPos.y + ImGui::GetFrameHeight());

                    ImGui::SetNextWindowPos(dropdownPos);
                    ImGui::SetNextWindowSize(ImVec2(dropdownWidth, dropdownHeight));

                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f * scalingFactor, 8.0f * scalingFactor));
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f * scalingFactor);
                    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 0.95f));

                    if (ImGui::Begin("##ModuleDropdown", nullptr, ImGuiWindowFlags_NoTitleBar |
                                                                  ImGuiWindowFlags_NoResize |
                                                                  ImGuiWindowFlags_NoMove |
                                                                  ImGuiWindowFlags_NoScrollbar |
                                                                  ImGuiWindowFlags_NoSavedSettings |
                                                                  ImGuiWindowFlags_NoFocusOnAppearing)) {

                        ImGui::BeginChild("ModuleList", ImVec2(0, 0), false, ImGuiWindowFlags_NoNavInputs);

                        if (inputChanged && !moduleSearchText.empty()) {
                            ImGui::SetScrollY(0);
                        }

                        const char* filterText = moduleSearchText.empty() ? "" : moduleSearchText.c_str();

                        int flatIndex = 0;
                        for (const auto& [id, module] : Store::BlockMetadataList(filterText)) {

                            ImGui::PushFont(styling.fonts.bold, 0.0f);
                            ImGui::TextUnformatted(module.title.c_str());
                            ImGui::PopFont();
                            ImGui::SameLine();
                            ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);

                            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
                            ImGui::SetNextWindowSize(ImVec2(600.0f * scalingFactor, 0.0f));
                            if (ImGui::BeginPopupContextItem(("fixed-block-description-" + id).c_str())) {
                                ImGui::TextWrapped(ICON_FA_BOOK " Description");
                                ImGui::Separator();
                                ImGui::Markdown(module.description.c_str(), module.description.length(), styling.config);
                                ImGui::EndPopup();
                            }
                            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                                ImGui::OpenPopupOnItemClick(("fixed-block-description-" + id).c_str(), ImGuiPopupFlags_MouseButtonLeft);
                                ImGui::SetNextWindowSize(ImVec2(600.0f * scalingFactor, 0.0f));
                                ImGui::BeginTooltip();
                                ImGui::TextWrapped(ICON_FA_BOOK " Description");
                                ImGui::SameLine();
                                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                                ImGui::Text("(click to pin)");
                                ImGui::PopStyleColor();
                                ImGui::Separator();
                                ImGui::Markdown(module.description.c_str(), module.description.length(), styling.config);
                                ImGui::EndTooltip();
                            }
                            ImGui::PopStyleVar();

                            ImGui::TextWrapped("%s", module.summary.c_str());

                            for (const auto& [device, _] : module.options) {
                                bool isSelected = (flatIndex == selectedModuleIndex);

                                U32 deviceColor;
                                switch (device) {
                                    case Device::CPU:
                                        deviceColor = CPUColor;
                                        break;
                                    case Device::CUDA:
                                        deviceColor = CUDAColor;
                                        break;
                                    case Device::Metal:
                                        deviceColor = MetalColor;
                                        break;
                                    case Device::Vulkan:
                                        deviceColor = VulkanColor;
                                        break;
                                    case Device::WebGPU:
                                        deviceColor = WebGPUColor;
                                        break;
                                    default:
                                        continue;
                                }

                                if (isSelected) {
                                    float margin = -1.0f * scalingFactor;
                                    ImVec2 pos = ImGui::GetCursorScreenPos();
                                    ImVec2 itemMin = ImVec2(pos.x - margin, pos.y - margin);
                                    ImVec2 itemMax = ImVec2(pos.x + margin + ImGui::CalcTextSize(ICON_FA_CUBE).x, pos.y + margin + ImGui::GetTextLineHeight());
                                    ImDrawList* drawList = ImGui::GetWindowDrawList();
                                    drawList->AddRectFilled(itemMin, itemMax, IM_COL32(255, 255, 255, 75), 3.0f);

                                    if (shouldScrollToSelected) {
                                        ImGui::SetScrollHereY(0.5f);
                                        shouldScrollToSelected = false;
                                    }
                                }

                                ImGui::PushStyleColor(ImGuiCol_Text, deviceColor);
                                ImGui::Text(ICON_FA_CUBE);
                                ImGui::PopStyleColor();

                                if (ImGui::IsItemClicked()) {
                                    moduleToAttach = {id, device};
                                    moduleAttachMode = true;
                                    showModuleDropdown = false;
                                    selectedModuleIndex = -1;
                                    moduleSearchText.clear();
                                }

                                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                                    ImGui::BeginTooltip();
                                    ImGui::Text("%s (%s)", module.title.c_str(), GetDevicePrettyName(device));
                                    ImGui::EndTooltip();
                                }

                                auto deviceIter = std::find_if(module.options.begin(), module.options.end(),
                                    [&device](const auto& pair) { return pair.first == device; });
                                if (deviceIter != module.options.end() && std::next(deviceIter) != module.options.end()) {
                                    ImGui::SameLine();
                                }

                                flatIndex++;
                            }

                            ImGui::Spacing();
                            ImGui::Separator();
                        }

                        const std::string text = "\n       " ICON_FA_HAND_SPOCK "\n\n-- END OF LIST --\n\n";
                        auto windowWidth = ImGui::GetWindowSize().x;
                        auto textWidth   = ImGui::CalcTextSize(text.c_str()).x;

                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                        ImGui::SetCursorPosX((windowWidth - textWidth) * 0.5f);
                        ImGui::TextUnformatted(text.c_str());
                        ImGui::PopStyleColor();

                        ImGui::EndChild();
                        ImGui::End();
                    }

                    ImGui::PopStyleColor();
                    ImGui::PopStyleVar(2);
                }
            }

            ImGui::PopStyleVar();
        }

        currentHeight += ImGui::GetWindowSize().y;
        ImGui::End();
    }

    //
    // Submit interactions.
    //

    // Save
    if (interactionTrigger == 1 && flowgraphLoaded) {
        if (state.flowgraphFilename.empty()) {
            state.globalModalToggle = true;
            state.globalModalContentId = 4;
        } else {
            mailboxes.saveFlowgraphMailbox = true;
        }
    }

    // New
    if (interactionTrigger == 2 && !flowgraphLoaded) {
        mailboxes.newFlowgraphMailbox = true;
    }

    // Open
    if (interactionTrigger == 3 && !flowgraphLoaded) {
        state.globalModalToggle = true;
        state.globalModalContentId = 3;
    }

    // Info
    if (interactionTrigger == 4 && flowgraphLoaded) {
        state.globalModalToggle = true;
        state.globalModalContentId = 4;
    }

    // Close
    if (interactionTrigger == 5 && flowgraphLoaded) {
        if (state.flowgraphFilename.empty() &&
            !instance.flowgraph().empty()) {
            state.globalModalToggle = true;
            state.globalModalContentId = 5;
        } else {
            mailboxes.closeFlowgraphMailbox = true;
        }
    }

    // Reset
    if (interactionTrigger == 6 && flowgraphLoaded) {
        mailboxes.resetFlowgraphMailbox = true;
    }

    // Source
    if (interactionTrigger == 7 && flowgraphLoaded) {
        state.sourceEditorEnabled = !state.sourceEditorEnabled;
        if (state.sourceEditorEnabled) {
            mailboxes.updateFlowgraphBlobMailbox = true;
        }
    }

    // Rename
    if (interactionTrigger == 8 && flowgraphLoaded) {
        state.globalModalToggle = true;
        state.globalModalContentId = 4;
    }

    // Exit fullscreen
    if (interactionTrigger == 9) {
        mailboxes.exitFullscreenMailbox = true;
    }

    //
    // Docking Arena.
    //

    const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                         ImGuiWindowFlags_NoNav |
                                         ImGuiWindowFlags_NoDocking |
                                         ImGuiWindowFlags_NoBackground |
                                         ImGuiWindowFlags_NoBringToFrontOnFocus |
                                         ImGuiWindowFlags_NoSavedSettings;

    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, currentHeight));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, viewport->Size.y - currentHeight));

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("##ArenaWindow", nullptr, windowFlags);
    ImGui::PopStyleVar();

    state.mainNodeId = ImGui::GetID("##Arena");
    ImGui::DockSpace(state.mainNodeId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode |
                                                     ImGuiDockNodeFlags_AutoHideTabBar);

    ImGui::End();

    //
    // Draw notifications.
    //

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, scalingFactor * 5.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(43.0f  / 255.0f,
                                                    43.0f  / 255.0f,
                                                    43.0f  / 255.0f,
                                                    100.0f / 255.0f));
    // ImGui::RenderNotifications();  // TODO: Re-enable when imgui-notify is available
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    //
    // Draw Source Editor.
    //

    [&](){
        if (!state.sourceEditorEnabled) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin("Source File", &state.sourceEditorEnabled)) {
            ImGui::End();
            return;
        }

        const bool editorFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
        const bool ctrlHeld = ImGui::GetIO().KeyCtrl;
        const bool shiftHeld = ImGui::GetIO().KeyShift;

        if (editorFocused && ctrlHeld) {
            if (ImGui::IsKeyPressed(ImGuiKey_S)) {
                mailboxes.applyFlowgraphMailbox = std::vector<char>(state.flowgraphSourceBuffer.begin(),
                                                          state.flowgraphSourceBuffer.end());
            } else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
                if (shiftHeld) {
                    mailboxes.updateFlowgraphBlobMailbox = true;
                } else if (state.flowgraphSourceDirty) {
                    state.flowgraphSourceDirty = false;
                    mailboxes.updateFlowgraphBlobMailbox = true;
                }
            }
        }

        if (state.flowgraphSourceBuffer.empty()) {
            ImGui::Text("Empty source file.");
        } else {
            ImGui::TextDisabled(state.flowgraphSourceDirty ? ICON_FA_CIRCLE_EXCLAMATION " Unsaved changes"
                                                     : ICON_FA_CIRCLE_CHECK " Up to date");
            ImGui::SameLine();
            ImGui::TextDisabled(ICON_FA_KEYBOARD " Ctrl+S apply, Ctrl+R revert, Ctrl+Shift+R reload");

            const ImVec2 editorSize = ImVec2(ImGui::GetContentRegionAvail().x,
                                             std::max(150.0f * scalingFactor,
                                                      ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing()));
            const ImGuiInputTextFlags editorFlags = ImGuiInputTextFlags_AllowTabInput |
                                                    ImGuiInputTextFlags_NoHorizontalScroll;
            if (ImGui::InputTextMultiline("##SourceFileData",
                                          &state.flowgraphSourceBuffer,
                                          editorSize,
                                          editorFlags)) {
                state.flowgraphSourceDirty = true;
            }

            ImGui::Spacing();

            if (state.flowgraphSourceDirty) {
                if (ImGui::Button("Apply Changes")) {
                    mailboxes.applyFlowgraphMailbox = std::vector<char>(state.flowgraphSourceBuffer.begin(),
                                                              state.flowgraphSourceBuffer.end());
                }
                ImGui::SameLine();
                if (ImGui::Button("Revert")) {
                    state.flowgraphSourceDirty = false;
                    mailboxes.updateFlowgraphBlobMailbox = true;
                }
            } else {
                if (ImGui::Button("Reload")) {
                    mailboxes.updateFlowgraphBlobMailbox = true;
                }
            }
        }

        ImGui::End();
    }();

    //
    // Draw graph.stacks.
    //

    std::vector<std::string> stacksToRemove;
    for (auto& [stack, state] : graph.stacks) {
        auto& [enabled, id] = state;

        if (!enabled) {
            stacksToRemove.push_back(stack);
            continue;
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowDockID(this->state.mainNodeId, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        ImGui::Begin(stack.c_str(), &enabled);
        ImGui::PopStyleVar();

        bool isDockNew = false;

        if (!id) {
            isDockNew = true;
            id = ImGui::GetID(jst::fmt::format("##Stack{}", stack).c_str());
        }

        ImGui::DockSpace(id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

        if (isDockNew && stack == "Graph") {
            ImGuiID dock_id_main;

            ImGui::DockBuilderRemoveNode(id);
            ImGui::DockBuilderAddNode(id);
            ImGui::DockBuilderSetNodePos(id, ImVec2(viewport->Pos.x, currentHeight));
            ImGui::DockBuilderSetNodeSize(id, ImVec2(viewport->Size.x, viewport->Size.y - currentHeight));

            dock_id_main = id;
            ImGui::DockBuilderDockWindow("Flowgraph", dock_id_main);

            ImGui::DockBuilderFinish(id);
        }

        ImGui::End();
    }
    for (const auto& stack : stacksToRemove) {
        graph.stacks.erase(stack);
    }

    //
    // Info HUD.
    //

    [&](){
        if (!state.infoPanelEnabled || state.fullscreenEnabled) {
            return;
        }

        const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                             ImGuiWindowFlags_NoDocking |
                                             ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoSavedSettings |
                                             ImGuiWindowFlags_NoFocusOnAppearing |
                                             ImGuiWindowFlags_NoNav |
                                             ImGuiWindowFlags_NoMove |
                                             ImGuiWindowFlags_Tooltip;

        const F32 windowPad = 10.0f * scalingFactor;
        ImVec2 workPos = viewport->WorkPos;
        ImVec2 workSize = viewport->WorkSize;
        ImVec2 windowPos, windowPosPivot;
        windowPos.x = workPos.x + windowPad;
        windowPos.y = workPos.y + workSize.y - windowPad;
        windowPosPivot.x = 0.0f;
        windowPosPivot.y = 1.0f;
        ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGui::SetNextWindowBgAlpha(0.35f);
        ImGui::Begin("Info HUD", nullptr, windowFlags);

        float fps = ImGui::GetIO().Framerate;
        if (fps > 50.0f) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
        }
        ImGui::TextFormatted("{:.0f} Hz", fps);
        if (fps > 50.0f) {
            ImGui::PopStyleColor();
        }
        ImGui::SameLine();
        ImGui::TextFormatted("{}", instance.viewport().name());
        instance.window().drawDebugMessage();

        ImGui::End();
    }();

    //
    // Fullscreen HUD.
    //

    [&](){
        if (!state.fullscreenEnabled) {
            return;
        }

        const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                             ImGuiWindowFlags_NoDocking |
                                             ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoSavedSettings |
                                             ImGuiWindowFlags_NoMove |
                                             ImGuiWindowFlags_Tooltip;

        const F32 windowPad = 12.0f * scalingFactor;
        ImVec2 workPos = viewport->WorkPos;
        ImVec2 workSize = viewport->WorkSize;
        ImVec2 windowPos, windowPosPivot;
        windowPos.x = workPos.x + workSize.x - windowPad;
        windowPos.y = windowPad;
        windowPosPivot.x = 1.0f;
        windowPosPivot.y = 0.0f;
        ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGui::SetNextWindowBgAlpha(0.5f);
        ImGui::Begin("Fullscreen HUD", nullptr, windowFlags);

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
        ImGui::TextUnformatted(ICON_FA_EXPAND " Fullscreen Mode (Press CTRL+L to exit)");
        ImGui::PopStyleColor();

        ImGui::End();
    }();

    //
    // Welcome HUD.
    //

    [&](){
        if (instance.flowgraph().created() || ImGui::IsPopupOpen("##help_modal")) {
            return;
        }

        const ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDecoration |
                                             ImGuiWindowFlags_NoDocking |
                                             ImGuiWindowFlags_AlwaysAutoResize |
                                             ImGuiWindowFlags_NoSavedSettings |
                                             ImGuiWindowFlags_NoFocusOnAppearing |
                                             ImGuiWindowFlags_NoNav |
                                             ImGuiWindowFlags_NoMove |
                                             ImGuiWindowFlags_Tooltip;

        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGui::SetNextWindowBgAlpha(0.35f);
        ImGui::Begin("Welcome", nullptr, windowFlags);

        ImGui::TextUnformatted(ICON_FA_USER_ASTRONAUT);
        ImGui::SameLine();
        ImGui::PushFont(styling.fonts.bold, 0.0f);
        ImGui::TextUnformatted("Welcome to CyberEther!");
        ImGui::PopFont();
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.6f));
        ImGui::TextFormatted("Version: {}", JETSTREAM_VERSION_STR);
        ImGui::PopStyleColor();

        ImGui::Separator();
        ImGui::Spacing();

        const char* largestText = "To get started, create a new flowgraph or open an existing one using";
        const auto largestTextSize = ImGui::CalcTextSize(largestText).x;

        static bool usePrimaryTexture = true;
        auto texture = usePrimaryTexture ? state.primaryBannerTexture : state.secondaryBannerTexture;
        const auto& [w, h] = texture->size();
        const auto ratio = static_cast<F32>(w) / static_cast<F32>(h);
        ImGui::Image(texture->raw(), ImVec2(largestTextSize, largestTextSize / ratio));

        if (ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            usePrimaryTexture = false;
        } else {
            usePrimaryTexture = true;
        }

        ImGui::Spacing();

        ImGui::Text("CyberEther is a tool designed for graphical visualization of radio");
        ImGui::Text("signals and general computing focusing in heterogeneous systems.");

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::TextUnformatted(largestText);
        ImGui::Text("the toolbar or the buttons below.");

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));

        if (ImGui::Button(ICON_FA_FILE " New Flowgraph")) {
            mailboxes.newFlowgraphMailbox = true;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_FOLDER_OPEN " Open Flowgraph")) {
            state.globalModalToggle = true;
            state.globalModalContentId = 3;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_VIAL " Open Examples")) {
            state.globalModalToggle = true;
            state.globalModalContentId = 3;
        }
        ImGui::SameLine();

        ImGui::PopStyleVar();

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::Text("To learn more about CyberEther, check the Help menu.");

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));

        if (ImGui::Button(ICON_FA_CIRCLE_QUESTION " Getting Started")) {
            state.globalModalToggle = true;
            state.globalModalContentId = 2;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CIRCLE_INFO " About CyberEther")) {
            state.globalModalToggle = true;
            state.globalModalContentId = 0;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CUBE " Block Backend Matrix")) {
            state.globalModalToggle = true;
            state.globalModalContentId = 1;
        }
        ImGui::SameLine();

        ImGui::PopStyleVar();

        ImGui::End();
    }();

    //
    // Global Modal
    //

    if (state.globalModalToggle) {
        ImGui::OpenPopup("##help_modal");
        if (state.globalModalContentId == 4) {
            state.filenameField = state.flowgraphFilename;
        }
        state.globalModalToggle = false;
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("##help_modal", nullptr, ImGuiWindowFlags_AlwaysAutoResize |
                                                        ImGuiWindowFlags_NoTitleBar |
                                                        ImGuiWindowFlags_NoResize |
                                                        ImGuiWindowFlags_NoMove|
                                                        ImGuiWindowFlags_NoScrollbar)) {
        if (state.globalModalContentId == 0) {
            ImGui::TextUnformatted(ICON_FA_CIRCLE_INFO " About CyberEther");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("CyberEther is a state-of-the-art tool designed for graphical visualization");
            ImGui::Text("of radio signals and computing focusing in heterogeneous systems.");

            ImGui::Spacing();

            ImGui::BulletText(ICON_FA_GAUGE_HIGH   " Graphical support for any device with Vulkan, Metal, or WebGPU.");
            ImGui::BulletText(ICON_FA_BATTERY_FULL " Portable GPU-acceleration for compute: NVIDIA (CUDA), Apple (Metal), etc.");
            ImGui::BulletText(ICON_FA_SHUFFLE      " Runtime flowgraph pipeline with heterogeneously-accelerated modular blocks.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextFormatted("Version: {}-{}", JETSTREAM_VERSION_STR, JETSTREAM_BUILD_TYPE);
            ImGui::Text("License: MIT License");
            ImGui::Text("Copyright (c) 2021-2024 Luigi F. Cruz");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 1) {
            ImGui::TextUnformatted(ICON_FA_CUBE " Block Backend Matrix");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("A CyberEther flowgraph is composed of modules, each module has a backend and a data type:");
            ImGui::BulletText("Backend is the hardware or software that will be used to process the data.");
            ImGui::BulletText("Data type is the format of the data that will be processed.");
            ImGui::Text("The following matrix shows the current installation available backends and data types.");

            ImGui::Spacing();

            const F32 screenHeight = io.DisplaySize.y;
            const ImVec2 tableHeight = ImVec2(0.0f, 0.60f * screenHeight);
            const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY;

            if (ImGui::BeginTable("table1", 5, tableFlags, tableHeight)) {
                static std::vector<std::tuple<const char*, float, U32, bool, Device>> columns = {
                    {"Block Name",  0.25f,   DefaultColor, false, Device::None},
                    {"CPU",         0.1875f, CPUColor,     true,  Device::CPU},
#if defined(JST_OS_MAC) || defined(JST_OS_IOS)
                    {"Metal",       0.1875f, MetalColor,   true,  Device::Metal},
#else
                    {"CUDA",        0.1875f, CUDAColor,  true,  Device::CUDA},
#endif
                    {"Vulkan",      0.1875f, VulkanColor,  true,  Device::Vulkan},
                    {"WebGPU",      0.1875f, WebGPUColor,  true,  Device::WebGPU}
                };

                ImGui::TableSetupScrollFreeze(0, 1);
                for (U64 col = 0; col < columns.size(); ++col) {
                    const auto& [name, width, _1, _2, _3] = columns[col];
                    ImGui::TableSetupColumn(name, ImGuiTableColumnFlags_WidthStretch, width);
                }

                ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
                for (U64 col = 0; col < columns.size(); ++col) {
                    const auto& [name, _1, color, cube, _2] = columns[col];
                    ImGui::TableSetColumnIndex(col);
                    if (cube) {
                        ImGui::PushStyleColor(ImGuiCol_Text, color);
                        ImGui::Text(ICON_FA_CUBE);
                        ImGui::PopStyleColor();
                        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                    }
                    ImGui::TableHeader(name);
                }

                for (const auto& [_, store] : Store::BlockMetadataList()) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(store.title.c_str());

                    for (U64 col = 1; col < 5; ++col) {
                        ImGui::TableSetColumnIndex(col);

                        const auto& device = std::get<4>(columns[col]);
                        if (!store.options.contains(device)) {
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, IM_COL32(255, 0, 0, 90));
                            continue;
                        }

                        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, IM_COL32(0, 255, 0, 30));
                        for (const auto& [inputDataType, outputDataType] : store.options.at(device)) {
                            const auto label = (outputDataType.empty()) ? jst::fmt::format("{}", inputDataType) :
                                                                          jst::fmt::format("{} -> {}", inputDataType, outputDataType);
                            ImGui::TextUnformatted(label.c_str());
                            ImGui::SameLine();
                        }
                    }
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 2) {
            ImGui::TextUnformatted(ICON_FA_CIRCLE_QUESTION " Getting started");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("To get started:");
            ImGui::BulletText("There is no start or stop button, the graph will run automatically.");
            ImGui::BulletText("Drag and drop a module from the Block Store to the Flowgraph.");
            ImGui::BulletText("Configure the module settings as needed.");
            ImGui::BulletText("To connect modules, drag from the output pin of one module to the input pin of another.");
            ImGui::BulletText("To disconnect modules, click on the input pin.");
            ImGui::BulletText("To remove a module, right click on it and select 'Remove Block'.");
            ImGui::Text("Ensure your device compatibility for optimal performance.");
            ImGui::Text("Need more help? Check the 'Help' section or visit our official documentation.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 3) {
            ImGui::TextUnformatted(ICON_FA_STORE " Flowgraph Store");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("This is the Flowgraph Store, a place where you can find and load flowgraphs.");
            ImGui::Text("You can also create your own flowgraphs and share them with the community.");
            ImGui::Text("To load a flowgraph, click in one of the bubbles below:");

            ImGui::Spacing();

            const F32 screenHeight = io.DisplaySize.y;
            const F32 maxTableHeight = 0.40f * screenHeight;

            const F32 lineHeight = ImGui::GetTextLineHeightWithSpacing();
            const F32 textPadding = lineHeight / 3.0f;
            const F32 rowHeight = lineHeight + (textPadding * 5.25f);
            const F32 totalTableHeight = rowHeight * std::ceil(Store::FlowgraphMetadataList().size() / 2.0f);

            const F32 tableHeight = (totalTableHeight < maxTableHeight) ? totalTableHeight : maxTableHeight;

            const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX |
                                               ImGuiTableFlags_NoBordersInBody |
                                               ImGuiTableFlags_NoBordersInBodyUntilResize |
                                               ImGuiTableFlags_ScrollY;

            if (ImGui::BeginTable("flowgraph_table", 2, tableFlags, ImVec2(0, tableHeight))) {
                for (U64 i = 0; i < 2; ++i) {
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0.5f);
                }

                U64 cellCount = 0;
                for (const auto& [id, flowgraph] : Store::FlowgraphMetadataList()) {
                    if ((cellCount % 2) == 0) {
                        ImGui::TableNextRow();
                    }
                    ImGui::TableSetColumnIndex(cellCount % 2);

                    const ImVec2 cellMin = ImGui::GetCursorScreenPos();
                    const ImVec2 cellSize = ImVec2(ImGui::GetColumnWidth(), lineHeight * 2 + textPadding);
                    const ImVec2 cellMax = ImVec2(cellMin.x + cellSize.x, cellMin.y + cellSize.y);

                    ImDrawList* drawList = ImGui::GetWindowDrawList();
                    drawList->AddRectFilled(cellMin, cellMax, IM_COL32(13, 13, 13, 138), 5.0f);

                    if (ImGui::InvisibleButton(("cell_button_" + id).c_str(), cellSize)) {
                        mailboxes.openFlowgraphBlobMailbox = flowgraph.data;
                        ImGui::CloseCurrentPopup();
                    }

                    ImGui::SetCursorScreenPos(ImVec2(cellMin.x + textPadding, cellMin.y + textPadding));
                    ImGui::PushFont(styling.fonts.h2, 0.0f);
                    ImGui::Text("%s", flowgraph.title.c_str());
                    ImGui::PopFont();
                    ImGui::SameLine();
                    ImGui::TextDisabled(ICON_FA_CIRCLE_QUESTION);
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                        ImGui::BeginTooltip();
                        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);

                        ImGui::TextWrapped("%s", flowgraph.description.c_str());

                        ImGui::PopTextWrapPos();
                        ImGui::EndTooltip();
                    }

                    ImGui::SetCursorScreenPos(ImVec2(cellMin.x + textPadding, cellMin.y + textPadding + lineHeight));
                    ImGui::Text("%s", flowgraph.summary.c_str());

                    cellCount += 1;
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();

            bool openFile = false;

            ImGui::Text(ICON_FA_FOLDER_OPEN " Or paste the path of a flowgraph file here:");
            static std::string globalModalPath;
            if (ImGui::BeginTable("flowgraph_table_path", 2, ImGuiTableFlags_NoBordersInBody |
                                                             ImGuiTableFlags_NoBordersInBodyUntilResize)) {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 80.0f);
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 20.0f);

                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::SetNextItemWidth(-1);
                if (ImGui::InputText("##FlowgraphPath", &globalModalPath, ImGuiInputTextFlags_EnterReturnsTrue)) {
                    openFile |= true;
                }

                ImGui::TableSetColumnIndex(1);
                if (ImGui::Button("Browse File", ImVec2(-1, 0))) {
                    const auto& res = Platform::PickFile(globalModalPath, {"yaml", "yml"});
                    if (res == Result::SUCCESS) {
                        openFile |= true;
                    }
                    JST_CHECK_NOTIFY(res);
                }

                ImGui::EndTable();
            }

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 6.0f, scalingFactor * 6.0f));
            openFile |= ImGui::Button(ICON_FA_PLAY " Load");
            if (openFile) {
                if (globalModalPath.size() == 0) {
//                     ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "Please enter a valid path or URL." });  // TODO: Re-enable when imgui-notify is available
                } else if (std::filesystem::exists(globalModalPath)) {
                    state.flowgraphFilename = globalModalPath;
                    mailboxes.openFlowgraphPathMailbox = true;
                    ImGui::CloseCurrentPopup();
                } else {
//                     ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "The specified path doesn't exist." });  // TODO: Re-enable when imgui-notify is available
                }
            }
            ImGui::PopStyleVar();

            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.6f));
            ImGui::SetCursorPosY(ImGui::GetCursorPosY());
            ImGui::Text("Make sure you trust the flowgraph source!");
            ImGui::PopStyleColor();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 4) {
            ImGui::TextUnformatted(ICON_FA_CIRCLE_INFO " Flowgraph Information");
            ImGui::Separator();
            ImGui::Spacing();

            const ImGuiTableFlags tableFlags = ImGuiTableFlags_PadOuterX;
            if (ImGui::BeginTable("##flowgraph-info-table", 2, tableFlags)) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TableSetupColumn("##flowgraph-info-table-labels", ImGuiTableColumnFlags_WidthStretch, 0.20f);
                ImGui::TableSetupColumn("##flowgraph-info-table-values", ImGuiTableColumnFlags_WidthStretch, 0.80f);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Title:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto title = instance.flowgraph().title();
                if (ImGui::InputText("##flowgraph-info-title", &title)) {
                    JST_CHECK_THROW(instance.flowgraph().setTitle(title));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Summary:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto summary = instance.flowgraph().summary();
                if (ImGui::InputText("##flowgraph-info-summary", &summary)) {
                    JST_CHECK_THROW(instance.flowgraph().setSummary(summary));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Author:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto author = instance.flowgraph().author();
                if (ImGui::InputText("##flowgraph-info-author", &author)) {
                    JST_CHECK_THROW(instance.flowgraph().setAuthor(author));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("License:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto license = instance.flowgraph().license();
                if (ImGui::InputText("##flowgraph-info-license", &license)) {
                    JST_CHECK_THROW(instance.flowgraph().setLicense(license));
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Description:");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                auto description = instance.flowgraph().description();
                const ImVec2 descSize = ImVec2(ImGui::GetContentRegionAvail().x,
                                               ImGui::GetTextLineHeightWithSpacing() * 6.0f);
                const ImGuiInputTextFlags descFlags = ImGuiInputTextFlags_AllowTabInput |
                                                      ImGuiInputTextFlags_NoHorizontalScroll;
                if (ImGui::InputTextMultiline("##flowgraph-info-description",
                                              &description,
                                              descSize,
                                              descFlags)) {
                    JST_CHECK_THROW(instance.flowgraph().setDescription(description));
                }

                ImGui::PopTextWrapPos();
                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 5.0f, scalingFactor * 5.0f));

            bool saveFile = false;

            if (ImGui::BeginTable("##flowgraph-info-path", 2, ImGuiTableFlags_NoBordersInBody |
                                                              ImGuiTableFlags_NoBordersInBodyUntilResize)) {
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 80.0f);
                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 20.0f);

                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::SetNextItemWidth(-1);

                if (ImGui::InputText("##flowgraph-info-filename", &state.filenameField, ImGuiInputTextFlags_EnterReturnsTrue)) {
                    saveFile |= true;
                }

                ImGui::TableSetColumnIndex(1);
                if (ImGui::Button("Browse File", ImVec2(-1, 0))) {
                    const auto& res = Platform::SaveFile(state.filenameField);
                    if (res == Result::SUCCESS) {
                        saveFile |= true;
                    }
                    JST_CHECK_NOTIFY(res);
                }

                ImGui::EndTable();
            }

            saveFile |= ImGui::Button(ICON_FA_FLOPPY_DISK " Save Flowgraph", ImVec2(-1, 0));
            if (saveFile) {
                [&]{
                    if (state.filenameField.empty()) {
                        JST_ERROR("[FLOWGRAPH] Filename is empty.");
                        JST_CHECK_NOTIFY(Result::ERROR);
                        return;
                    }

                    std::regex filenamePattern("^.+\\.ya?ml$");
                    if (!std::regex_match(state.filenameField, filenamePattern)) {
                        JST_ERROR("[FLOWGRAPH] Invalid filename '{}'.", state.filenameField);
                        JST_CHECK_NOTIFY(Result::ERROR);
                        return;
                    }

                    state.flowgraphFilename = state.filenameField;
                    mailboxes.saveFlowgraphMailbox = true;
                    ImGui::CloseCurrentPopup();
                }();
            }

            ImGui::PopStyleVar();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 5) {
            ImGui::TextUnformatted(ICON_FA_TRIANGLE_EXCLAMATION " Close Flowgraph");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("You are about to close a flowgraph without saving it.");
            ImGui::Text("Are you sure you want to continue?");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Save", ImVec2(-1, 0))) {
                state.globalModalContentId = 4;
            }
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
            if (ImGui::Button("Close Anyway", ImVec2(-1, 0))) {
                mailboxes.closeFlowgraphMailbox = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor();
            if (ImGui::Button("Cancel", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 6) {
            ImGui::TextUnformatted(ICON_FA_PENCIL " Rename Block");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##rename-block-new-id", &state.renameBlockNewId);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 1.0f, 1.0f));
            if (ImGui::Button("Rename Block", ImVec2(-1, 0))) {
                mailboxes.renameBlockMailbox = {state.renameBlockLocale, state.renameBlockNewId};
                ImGui::CloseCurrentPopup();
            }
            ImGui::PopStyleColor();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 7) {
            ImGui::TextUnformatted(ICON_FA_KEY " CyberEther License");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("MIT License");

            ImGui::Spacing();

            ImGui::Text("Copyright (c) 2021-2024 Luigi F. Cruz");

            ImGui::Spacing();

            ImGui::Text("Permission is hereby granted, free of charge, to any person obtaining a copy");
            ImGui::Text("of this software and associated documentation files (the \"Software\"), to deal");
            ImGui::Text("in the Software without restriction, including without limitation the rights");
            ImGui::Text("to use, copy, modify, merge, publish, distribute, sublicense, and/or sell");
            ImGui::Text("copies of the Software, and to permit persons to whom the Software is");
            ImGui::Text("furnished to do so, subject to the following conditions:");

            ImGui::Spacing();

            ImGui::Text("The above copyright notice and this permission notice shall be");
            ImGui::Text("included in all copies or substantial portions of the Software.");

            ImGui::Spacing();

            ImGui::Text("THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR");
            ImGui::Text("IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,");
            ImGui::Text("FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE");
            ImGui::Text("AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER");
            ImGui::Text("LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,");
            ImGui::Text("OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE");
            ImGui::Text("SOFTWARE.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("View Third-Party Licenses")) {
                Platform::OpenUrl("https://github.com/luigifcruz/CyberEther/blob/main/ACKNOWLEDGMENTS.md");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 8) {
            ImGui::TextUnformatted(ICON_FA_BOX_OPEN " Third-Party OSS");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("CyberEther utilizes the following open-source third-party software,");
            ImGui::Text("and we extend our gratitude to the creators of these libraries for");
            ImGui::Text("their valuable contributions to the open-source community.");

            ImGui::BulletText("Miniaudio - MIT License");
            ImGui::BulletText("Dear ImGui - MIT License");
            ImGui::BulletText("ImNodes - MIT License");
            ImGui::BulletText("PocketFFT - BSD-3-Clause License");
            ImGui::BulletText("RapidYAML - MIT License");
            ImGui::BulletText("vkFFT - MIT License");
            ImGui::BulletText("stb - MIT License");
            ImGui::BulletText("fmtlib - MIT License");
            ImGui::BulletText("SoapySDR - Boost Software License");
            ImGui::BulletText("GLFW - zlib/libpng License");
            ImGui::BulletText("imgui-notify - MIT License");
            ImGui::BulletText("spirv-cross - MIT License");
            ImGui::BulletText("glslang - BSD-3-Clause License");
            ImGui::BulletText("naga - Apache License 2.0");
            ImGui::BulletText("gstreamer - LGPL-2.1 License");
            ImGui::BulletText("libusb - LGPL-2.1 License");
            ImGui::BulletText("nanobench - MIT License");
            ImGui::BulletText("Catch2 - Boost Software License");
            ImGui::BulletText("JetBrains Mono - SIL Open Font License 1.1");
            ImGui::BulletText("imgui_markdown - Zlib License");
            ImGui::BulletText("GLM - Happy Bunny License");
            ImGui::BulletText("cpp-httplib - MIT License");
            ImGui::BulletText("nlohmann/json - MIT License");
            ImGui::BulletText("FTXUI - MIT License");
            // [NEW DEPENDENCY HOOK]

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("View Third-Party Licenses")) {
                Platform::OpenUrl("https://github.com/luigifcruz/CyberEther/blob/main/ACKNOWLEDGMENTS.md");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        } else if (state.globalModalContentId == 9) {
            ImGui::TextUnformatted(ICON_FA_GAUGE_HIGH " Benchmarking Tool");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("This is the Benchmarking Tool, a place where you can run benchmarks");
            ImGui::Text("to compare the performance between different devices and backends.");

            static std::ostringstream benchmarkData;
            static std::string buildInfoStr = jst::fmt::format("V{} ({}) - Optimization: {} - Debug: {} - Native: {}", JETSTREAM_VERSION_STR,
                                                                                                                  JETSTREAM_BUILD_TYPE,
                                                                                                                  JETSTREAM_BUILD_OPTIMIZATION,
                                                                                                                  JETSTREAM_BUILD_DEBUG,
                                                                                                                  JETSTREAM_BUILD_NATIVE);

            if (ImGui::Button("Run Benchmark")) {
                if (state.benchmarkRunning) {
//                     ImGui::InsertNotification({ ImGuiToastType_Error, 5000, "A benchmark is already state.running." });  // TODO: Re-enable when imgui-notify is available
                } else {
                    std::thread([&]{
                        benchmarkData.clear();
                        state.benchmarkRunning = true;
                        Benchmark::ResetResults();
//                         ImGui::InsertNotification({ ImGuiToastType_Info, 5000, "Running benchmark..." });  // TODO: Re-enable when imgui-notify is available
                        Benchmark::Run("markdown", benchmarkData);
                        state.benchmarkRunning = false;
                    }).detach();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Benchmark")) {
                if (!state.benchmarkRunning) {
                    benchmarkData.clear();
                    Benchmark::ResetResults();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Copy Benchmark Results")) {
                if (!state.benchmarkRunning) {
                    ImGui::SetClipboardText(jst::fmt::format("{}\n{}", buildInfoStr, benchmarkData.str()).c_str());
//                     ImGui::InsertNotification({ ImGuiToastType_Info, 5000, "Benchmark results copied to clipboard." });  // TODO: Re-enable when imgui-notify is available
                }
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            const auto& results = Benchmark::GetResults();

            const ImVec2 tableHeight = ImVec2(0.50f * io.DisplaySize.x, 0.50f * io.DisplaySize.y);
            const ImGuiTableFlags mainTableFlags = ImGuiTableFlags_PadOuterX |
                                                   ImGuiTableFlags_Borders |
                                                   ImGuiTableFlags_ScrollY |
                                                   ImGuiTableFlags_Reorderable |
                                                   ImGuiTableFlags_Hideable;

            if (ImGui::BeginTable("benchmark-table", 2, mainTableFlags, tableHeight)) {
                ImGui::TableSetupColumn("Module", ImGuiTableColumnFlags_WidthStretch, 0.175f);
                ImGui::TableSetupColumn("Results", ImGuiTableColumnFlags_WidthStretch, 0.825f);
                ImGui::TableHeadersRow();

                for (const auto& [name, entries] : results) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(name.c_str());

                    const ImGuiTableFlags nestedTableFlags = ImGuiTableFlags_Borders |
                                                             ImGuiTableFlags_Reorderable |
                                                             ImGuiTableFlags_Hideable;

                    ImGui::TableNextColumn();
                    if (ImGui::BeginTable(("benchmark-subtable-" + name).c_str(), 4, nestedTableFlags)) {
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 0.60f);
                        ImGui::TableSetupColumn("ms/op", ImGuiTableColumnFlags_WidthStretch, 0.10f);
                        ImGui::TableSetupColumn("op/s", ImGuiTableColumnFlags_WidthStretch, 0.20f);
                        ImGui::TableSetupColumn("err%", ImGuiTableColumnFlags_WidthStretch, 0.10f);
                        ImGui::TableHeadersRow();

                        for (const auto& entry : entries) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", entry.name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::Text("%.2f", entry.ms_per_op);
                            ImGui::TableNextColumn();
                            ImGui::Text("%.2f", entry.ops_per_sec);
                            ImGui::TableNextColumn();
                            ImGui::Text("%.2f", entry.error);
                        }

                        ImGui::EndTable();
                    }
                }

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("Binary Information:");
            ImGui::TextUnformatted(buildInfoStr.c_str());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            std::string progressTaint;
            if (Benchmark::TotalCount() == Benchmark::CurrentCount()) {
                progressTaint = "COMPLETE";
            } else {
                progressTaint = jst::fmt::format("{}/{}", Benchmark::CurrentCount(), Benchmark::TotalCount());
            }
            ImGui::TextFormatted("Benchmark Progress [{}]", progressTaint);
            F32 progress =  Benchmark::TotalCount() > 0 ? static_cast<F32>(Benchmark::CurrentCount()) /  Benchmark::TotalCount() : 0.0f;
            ImGui::ProgressBar(progress, ImVec2(-1, 0), "");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            if (ImGui::Button("Close", ImVec2(-1, 0))) {
                ImGui::CloseCurrentPopup();
            }
        }

        if (interactionTrigger == 99) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::SetItemDefaultFocus();
        ImGui::Dummy(ImVec2(550.0f * scalingFactor, 0.0f));
        ImGui::EndPopup();
    }

    //
    // Debug Latency Render
    //

    [&](){
        if (!state.debugLatencyEnabled) {
            return;
        }

        const auto& mainWindowWidth = io.DisplaySize.x;
        const auto& mainWindowHeight = io.DisplaySize.y;

        const auto timerWindowWidth  = 200.0f * scalingFactor;
        const auto timerWindowHeight = 85.0f * scalingFactor;

        static F32 x = 0.0f;
        static F32 xd = 1.0f;

        x += xd;

        if (x > (mainWindowWidth - timerWindowWidth)) {
            xd = -xd;
        }
        if (x < 0.0f) {
            xd = -xd;
        }

        ImGui::SetNextWindowSize(ImVec2(timerWindowWidth, timerWindowHeight));
        ImGui::SetNextWindowPos(ImVec2(x, (mainWindowHeight / 2.0f) - (timerWindowHeight / 2.0f)));

        if (!ImGui::Begin("Lantency Debug", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse)) {
            ImGui::End();
            return;
        }

        U64 ms = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        ImGui::TextFormatted("Time: {} ms", ms);

        const F32 blockWidth = timerWindowWidth / 6.0f;
        const F32 blockHeight = 35.0f;
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        ImVec2 blockPos = ImVec2(windowPos.x, windowPos.y + timerWindowHeight - blockHeight);
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(255, 0, 0, 255));      // Red
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(0, 255, 0, 255));      // Green
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(0, 0, 255, 255));      // Blue
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(255, 255, 0, 255));    // Yellow
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(255, 255, 255, 255));  // White
        blockPos.x += blockWidth;
        drawList->AddRectFilled(blockPos, ImVec2(blockPos.x + blockWidth, blockPos.y + blockHeight), IM_COL32(0, 0, 0, 255));        // Black

        ImGui::End();
    }();

    //
    // Debug Viewport Render
    //

    [&](){
        if (!state.debugViewportEnabled) {
            return;
        }

        if (!ImGui::Begin("Viewport Debug")) {
            ImGui::End();
            return;
        }

        // Report the window size.

        const auto& mainWindowWidth = io.DisplaySize.x;
        const auto& mainWindowHeight = io.DisplaySize.y;
        ImGui::Text("Window Size: %.2f x %.2f", mainWindowWidth, mainWindowHeight);

        // Report the framebuffer scale.

        const auto& framebufferWidth = io.DisplayFramebufferScale.x;
        const auto& framebufferHeight = io.DisplayFramebufferScale.y;
        ImGui::Text("Framebuffer Scale: %.2f x %.2f", framebufferWidth, framebufferHeight);

        // Render window scale.

        const auto& windowWidth = instance.window().scalingFactor();
        ImGui::Text("Render Window Scale: %.2f", windowWidth);

        ImGui::End();
    }();

    //
    // Debug Demo Render
    //

    if (state.debugDemoEnabled) {
        ImGui::ShowDemoWindow();
    }

    return Result::SUCCESS;
}

Result Compositor::drawFlowgraph() {
    // Load local variables.
    const auto& scalingFactor = instance.window().scalingFactor();
    const auto& nodeStyle = ImNodes::GetStyle();
    const auto& guiStyle = ImGui::GetStyle();
    const auto& io = ImGui::GetIO();

    const F32 windowMinWidth = 300.0f * scalingFactor;
    const F32 variableWidth = 100.0f * scalingFactor;

    //
    // View Render
    //

    for (const auto& [_, state] : graph.nodeStates) {
        if (!state.enabled ||
            !state.block->getState().viewEnabled ||
            !state.block->shouldDrawView() ||
            !state.block->complete() ||
            this->state.fullscreenEnabled) {
            continue;
        }

        ImGui::SetNextWindowSizeConstraints(ImVec2(64.0f, 64.0f),
                                            ImVec2(io.DisplaySize.x, io.DisplaySize.y));
        ImGui::SetNextWindowSize(ImVec2(400.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin(jst::fmt::format("View - {}", state.title).c_str(),
                          &state.block->state.viewEnabled)) {
            ImGui::End();
            continue;
        }
        state.block->drawView();
        ImGui::End();
    }

    //
    // Fullscreen Render
    //

    for (const auto& [_, state] : graph.nodeStates) {
        if (!state.enabled ||
            !state.block->getState().fullscreenEnabled ||
            !state.block->shouldDrawFullscreen() ||
            !state.block->complete()) {
            this->state.fullscreenEnabled = false;
            continue;
        }

        if (mailboxes.exitFullscreenMailbox.has_value()) {
            state.block->state.fullscreenEnabled = false;
            mailboxes.exitFullscreenMailbox.reset();
            this->state.fullscreenEnabled = false;
            break;
        }

        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);

        const auto flags = ImGuiWindowFlags_NoTitleBar |
                           ImGuiWindowFlags_NoResize |
                           ImGuiWindowFlags_NoMove |
                           ImGuiWindowFlags_NoScrollbar |
                           ImGuiWindowFlags_NoScrollWithMouse |
                           ImGuiWindowFlags_NoCollapse |
                           ImGuiWindowFlags_NoSavedSettings;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        if (!ImGui::Begin(jst::fmt::format("Fullscreen - {}", state.title).c_str(), &state.block->state.viewEnabled, flags)) {
            ImGui::End();
            ImGui::PopStyleVar();
            ImGui::PopStyleVar();
            continue;
        }
        state.block->drawView();
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();

        this->state.fullscreenEnabled = true;
        break;
    }

    //
    // Control Render
    //

    for (const auto& [_, state] : graph.nodeStates) {
        if (!state.enabled ||
            !state.block->getState().controlEnabled ||
            !state.block->shouldDrawControl() ||
            this->state.fullscreenEnabled) {
            continue;
        }

        ImGui::SetNextWindowSizeConstraints(ImVec2(64.0f, 64.0f),
                                            ImVec2(io.DisplaySize.x, io.DisplaySize.y));
        ImGui::SetNextWindowSize(ImVec2(400.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin(jst::fmt::format("Control - {}", state.title).c_str(),
                          &state.block->state.controlEnabled)) {
            ImGui::End();
            continue;
        }

        ImGui::BeginTable("##ControlTable", 2, ImGuiTableFlags_None);
        ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
        ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, ImGui::GetWindowWidth() - variableWidth -
                                                                             (guiStyle.CellPadding.x * 2.0f));
        state.block->drawControl();
        ImGui::EndTable();

        if (state.block->shouldDrawInfo()) {
            if (ImGui::TreeNode("Info")) {
                ImGui::BeginTable("##InfoTableAttached", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthFixed, ImGui::GetWindowWidth() - variableWidth -
                                                                                  (guiStyle.CellPadding.x * 2.0f));
                state.block->drawInfo();
                ImGui::EndTable();
                ImGui::TreePop();
            }
        }

        ImGui::Dummy(ImVec2(windowMinWidth, 0.0f));

        ImGui::End();
    }

    //
    // Flowgraph Render
    //

    [&](){
        if (!state.flowgraphEnabled || state.fullscreenEnabled || !instance.flowgraph().created()) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(500.0f * scalingFactor, 300.0f * scalingFactor), ImGuiCond_FirstUseEver);

        if (!ImGui::Begin("Flowgraph")) {
            ImGui::End();
            return;
        }

        // Set node position according to the internal state.
        for (const auto& [locale, state] : graph.nodeStates) {
            const auto& [x, y] = state.block->getState().nodePos;
            ImNodes::SetNodeGridSpacePos(state.id, ImVec2(x * scalingFactor, y * scalingFactor));
        }

        ImNodes::BeginNodeEditor();
        ImNodes::MiniMap(0.075f * scalingFactor, ImNodesMiniMapLocation_TopRight);

        for (const auto& [locale, state] : graph.nodeStates) {
            const auto& block = state.block;
            const auto& moduleEntry = Store::BlockMetadataList().at(block->id());

            F32 nodeWidth = block->state.nodeWidth * scalingFactor;
            const F32 titleWidth = ImGui::CalcTextSize(state.title.c_str()).x +
                                   ImGui::CalcTextSize(" " ICON_FA_CIRCLE_QUESTION).x +
                                   ((!block->complete()) ?
                                        ImGui::CalcTextSize(" " ICON_FA_SKULL).x : 0) +
                                   ((!block->warning().empty() && block->complete()) ?
                                        ImGui::CalcTextSize(" " ICON_FA_TRIANGLE_EXCLAMATION).x : 0);
            const F32 controlWidth = block->shouldDrawControl() ? windowMinWidth: 0.0f;
            const F32 previewWidth = block->shouldDrawPreview() ? windowMinWidth : 0.0f;
            nodeWidth = std::max({titleWidth, nodeWidth, controlWidth, previewWidth});

            // Push node-specific style.
            const bool nodeEnabled = state.enabled;
            if (block->complete() && nodeEnabled) {
                switch (block->device()) {
                    case Device::CPU:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         CPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  CPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, CPUColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              CPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       CPUColorSelected);
                        break;
                    case Device::CUDA:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         CUDAColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  CUDAColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, CUDAColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              CUDAColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       CUDAColorSelected);
                        break;
                    case Device::Metal:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, MetalColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       MetalColorSelected);
                        break;
                    case Device::Vulkan:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, VulkanColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       VulkanColorSelected);
                        break;
                    case Device::WebGPU:
                        ImNodes::PushColorStyle(ImNodesCol_TitleBar,         WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, WebGPUColorSelected);
                        ImNodes::PushColorStyle(ImNodesCol_Pin,              WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_PinHovered,       WebGPUColorSelected);
                        break;
                    case Device::None:
                        break;
                }
            } else {
                ImNodes::PushColorStyle(ImNodesCol_TitleBar,         DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered,  DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, DisabledColorSelected);
                ImNodes::PushColorStyle(ImNodesCol_Pin,              DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_PinHovered,       DisabledColorSelected);
            }

            ImNodes::BeginNode(state.id);

            // Draw node title.
            ImNodes::BeginNodeTitleBar();

            ImGui::TextUnformatted(state.title.c_str());

            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
            ImGui::Text(ICON_FA_CIRCLE_QUESTION);
            ImGui::PopStyleColor();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
            ImGui::SetNextWindowSize(ImVec2(600.0f * scalingFactor, 0.0f));
            if (ImGui::BeginPopupContextItem("fixed-block-description")) {
                ImGui::TextWrapped(ICON_FA_BOOK " Description");
                ImGui::Separator();
                ImGui::Markdown(moduleEntry.description.c_str(), moduleEntry.description.length(), styling.config);
                ImGui::EndPopup();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                ImGui::OpenPopupOnItemClick("fixed-block-description", ImGuiPopupFlags_MouseButtonLeft);
                ImGui::SetNextWindowSize(ImVec2(600.0f * scalingFactor, 0.0f));
                ImGui::BeginTooltip();
                ImGui::TextWrapped(ICON_FA_BOOK " Description");
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                ImGui::Text("(click to pin)");
                ImGui::PopStyleColor();
                ImGui::Separator();
                ImGui::Markdown(moduleEntry.description.c_str(), moduleEntry.description.length(), styling.config);
                ImGui::EndTooltip();
            }
            ImGui::PopStyleVar();

            if (!block->complete()) {
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                ImGui::Text(ICON_FA_SKULL);
                ImGui::PopStyleColor();
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextWrapped(ICON_FA_SKULL " Error Message");
                    ImGui::Separator();
                    ImGui::TextWrapped("%s", block->error().c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
                ImGui::PopStyleVar();
            } else if (!block->warning().empty()) {
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
                ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION);
                ImGui::PopStyleColor();
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(scalingFactor * 8.0f, scalingFactor * 8.0f));
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextWrapped(ICON_FA_TRIANGLE_EXCLAMATION " Warning Message");
                    ImGui::Separator();
                    ImGui::TextWrapped("%s", block->warning().c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
                ImGui::PopStyleVar();
            }

            ImNodes::EndNodeTitleBar();

            // Suspend ImNodes default styling.
            ImGui::PopStyleVar();
            ImGui::PopStyleVar();

            // Draw node info.
            if (block->shouldDrawInfo()) {
                ImGui::BeginTable("##NodeInfoTable", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthFixed, nodeWidth -  variableWidth -
                                                                                  (guiStyle.CellPadding.x * 2.0f));
                block->drawInfo();
                ImGui::EndTable();
            }

            // Draw node control.
            if (block->shouldDrawControl()) {
                ImGui::BeginTable("##NodeControlTable", 2, ImGuiTableFlags_None);
                ImGui::TableSetupColumn("Variable", ImGuiTableColumnFlags_WidthFixed, variableWidth);
                ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed, nodeWidth -  variableWidth -
                                                                                     (guiStyle.CellPadding.x * 2.0f));
                block->drawControl();
                ImGui::EndTable();
            }

            // Restore ImNodes default styling.
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(1.0f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            // Draw node input and output pins.
            if (!state.inputs.empty() || !state.outputs.empty()) {
                ImGui::Spacing();

                ImNodes::PushAttributeFlag(ImNodesAttributeFlags_EnableLinkDetachWithDragClick);
                for (const auto& [inputPinId, _] : state.inputs) {
                    const auto& pinName = graph.pinLocaleMap.at(inputPinId).pinId;

                    ImNodes::BeginInputAttribute(inputPinId);
                    ImGui::TextUnformatted(pinName.c_str());
                    ImNodes::EndInputAttribute();
                }
                ImNodes::PopAttributeFlag();

                for (const auto& [outputPinId, _] : state.outputs) {
                    const auto& pinName = graph.pinLocaleMap.at(outputPinId).pinId;
                    const F32 textWidth = ImGui::CalcTextSize(pinName.c_str()).x;

                    ImNodes::BeginOutputAttribute(outputPinId);
                    ImGui::Indent(nodeWidth - textWidth);
                    ImGui::TextUnformatted(pinName.c_str());
                    ImNodes::EndOutputAttribute();
                }
            }

            // Draw node preview.
            if (block->complete() &&
                block->shouldDrawPreview() &&
                block->getState().previewEnabled) {
                ImGui::Spacing();
                block->drawPreview(nodeWidth);
            }

            // Ensure minimum width set by the internal state.
            ImGui::Dummy(ImVec2(nodeWidth, 0.0f));

            // Draw interfacing options.
            if (block->shouldDrawView()    ||
                block->shouldDrawPreview() ||
                block->shouldDrawControl() ||
                block->shouldDrawInfo()) {
                ImGui::BeginTable("##NodeInterfacingOptionsTable", 3, ImGuiTableFlags_None, ImVec2(nodeWidth, 0.0f));
                const F32 buttonSize = 25.0f * scalingFactor;
                ImGui::TableSetupColumn("Switches", ImGuiTableColumnFlags_WidthFixed, nodeWidth - (buttonSize * 2.0f) -
                                                                                      (guiStyle.CellPadding.x * 4.0f));
                ImGui::TableSetupColumn("Minus", ImGuiTableColumnFlags_WidthFixed, buttonSize);
                ImGui::TableSetupColumn("Plus", ImGuiTableColumnFlags_WidthFixed, buttonSize);
                ImGui::TableNextRow();

                // Switches
                ImGui::TableSetColumnIndex(0);

                if (block->shouldDrawView()) {
                    ImGui::Checkbox("Window", &block->state.viewEnabled);

                    if (block->shouldDrawControl() ||
                        block->shouldDrawInfo()    ||
                        block->shouldDrawPreview() ||
                        block->shouldDrawFullscreen()) {
                        ImGui::SameLine();
                    }
                }

                if (block->shouldDrawFullscreen()) {
                    ImGui::Checkbox("Fullscreen", &block->state.fullscreenEnabled);

                    if (block->shouldDrawControl() ||
                        block->shouldDrawInfo()    ||
                        block->shouldDrawPreview()) {
                        ImGui::SameLine();
                    }
                }

                if (block->shouldDrawControl() ||
                    block->shouldDrawInfo()) {
                    ImGui::Checkbox("Control", &block->state.controlEnabled);

                    if (block->shouldDrawPreview()) {
                        ImGui::SameLine();
                    }
                }

                if (block->shouldDrawPreview()) {
                    ImGui::Checkbox("Preview", &block->state.previewEnabled);
                }

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scalingFactor * 2.0f, scalingFactor * 1.0f));

                // Minus Button
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                if (ImGui::Button(" - ")) {
                    nodeWidth -= 25.0f * scalingFactor;
                }

                // Plus Button
                ImGui::TableSetColumnIndex(2);
                ImGui::SetNextItemWidth(-1);
                if (ImGui::Button(" + ")) {
                    nodeWidth += 25.0f * scalingFactor;
                }

                ImGui::PopStyleVar();

                ImGui::EndTable();
            }

            ImNodes::EndNode();

            // Pop node-specific style.
            ImNodes::PopColorStyle(); // TitleBar
            ImNodes::PopColorStyle(); // TitleBarHovered
            ImNodes::PopColorStyle(); // TitleBarSelected
            ImNodes::PopColorStyle(); // Pin
            ImNodes::PopColorStyle(); // PinHovered

            // Update node width.
            block->state.nodeWidth = nodeWidth / scalingFactor;
        }

        // Draw node links.
        for (const auto& [linkId, locales] : graph.linkLocaleMap) {
            const auto& [inputLocale, outputLocale] = locales;
            const auto& inputPinId = graph.inputLocalePinMap.at(inputLocale);
            const auto& outputPinId = graph.outputLocalePinMap.at(outputLocale);
            const auto& outputNodeState = graph.nodeStates.at(outputLocale.block());
            const auto& outputBlock = outputNodeState.block;

            if (outputBlock->complete() && outputNodeState.enabled) {
                switch (outputBlock->device()) {
                    case Device::CPU:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         CPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  CPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, CPUColorSelected);
                        break;
                    case Device::CUDA:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         CUDAColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  CUDAColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, CUDAColorSelected);
                        break;
                    case Device::Metal:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  MetalColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, MetalColorSelected);
                        break;
                    case Device::Vulkan:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  VulkanColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, VulkanColorSelected);
                        break;
                    case Device::WebGPU:
                        ImNodes::PushColorStyle(ImNodesCol_Link,         WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  WebGPUColor);
                        ImNodes::PushColorStyle(ImNodesCol_LinkSelected, WebGPUColorSelected);
                        break;
                    case Device::None:
                        break;
                }
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Link,         DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_LinkHovered,  DisabledColor);
                ImNodes::PushColorStyle(ImNodesCol_LinkSelected, DisabledColorSelected);
            }

            ImNodes::Link(linkId, inputPinId, outputPinId);

            ImNodes::PopColorStyle(); // Link
            ImNodes::PopColorStyle(); // LinkHovered
            ImNodes::PopColorStyle(); // LinkSelected
        }

        ImNodes::EndNodeEditor();

        // Update internal state node position.
        for (const auto& [locale, state] : graph.nodeStates) {
            const auto& [x, y] = ImNodes::GetNodeGridSpacePos(state.id);
            state.block->state.nodePos = {x / scalingFactor, y / scalingFactor};
        }

        // Spatially organize graph.
        if (!state.graphSpatiallyOrganized) {
            JST_DEBUG("[COMPOSITOR] Running graph auto-route.");

            F32 previousClustersHeight = 0.0f;

            for (const auto& cluster : graph.nodeTopology) {
                F32 largestColumnHeight = 0.0f;
                F32 previousColumnsWidth = 0.0f;

                for (const auto& column : cluster) {
                    F32 largestNodeWidth = 0.0f;
                    F32 previousNodesHeight = 0.0f;

                    for (const auto& nodeId : column) {
                        auto dims = ImNodes::GetNodeDimensions(nodeId);

                        // Unpack dimensions.
                        dims.x /= scalingFactor;
                        dims.y /= scalingFactor;

                        auto& block = graph.nodeStates.at(graph.nodeLocaleMap.at(nodeId)).block;
                        auto& [x, y] = block->state.nodePos;

                        // Add previous columns horizontal offset.
                        x = previousColumnsWidth;
                        // Add previous clusters and rows vertical offset.
                        y = previousNodesHeight + previousClustersHeight;

                        previousNodesHeight += dims.y + 25.0f;
                        largestNodeWidth = std::max({
                            dims.x,
                            largestNodeWidth,
                        });

                        // Add left padding to nodes in the same column.
                        x += (largestNodeWidth - dims.x);
                    }

                    largestColumnHeight = std::max({
                        previousNodesHeight,
                        largestColumnHeight,
                    });

                    previousColumnsWidth += largestNodeWidth + 37.5f;
                }

                previousClustersHeight += largestColumnHeight + 12.5;
            }

            ImNodes::EditorContextResetPanning({0.0f, 0.0f});
            state.graphSpatiallyOrganized = true;
        }

        // Render underlying buffer information about the link.
        I32 linkId;
        if (ImNodes::IsLinkHovered(&linkId)) {
            const auto& [inputLocale, outputLocale] = graph.linkLocaleMap.at(linkId);
            const auto& rec = graph.nodeStates.at(outputLocale.block()).outputMap.at(outputLocale.pinId);

            ImGui::BeginTooltip();
            ImGui::TextWrapped(ICON_FA_MEMORY " Tensor Metadata");
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
            ImGui::Text(ICON_FA_CIRCLE_INFO " Click on the end of the link to detach it.");
            ImGui::PopStyleColor();
            ImGui::Separator();

            const auto firstLine  = jst::fmt::format("[{} -> {}]", outputLocale, inputLocale);
            const auto secondLine = jst::fmt::format("[{}] {} [{}] [Device::{}]", rec.dataType, rec.shape, rec.contiguous ? "C" : "NC", rec.device);
            const auto thirdLine  = jst::fmt::format("[Host Acessible: {}] [Device Native: {}] [Host Native: {}]", rec.host_accessible ? "YES" : "NO",
                                                                                                                   rec.device_native ? "YES" : "NO",
                                                                                                                   rec.host_native ? "YES" : "NO");
            const auto forthLine  = jst::fmt::format("[PTR: 0x{:016X}] [HASH: 0x{:016X}]", reinterpret_cast<uintptr_t>(rec.data), rec.hash);

            ImGui::TextUnformatted(firstLine.c_str());
            ImGui::TextUnformatted(secondLine.c_str());
            ImGui::TextUnformatted(thirdLine.c_str());
            ImGui::TextUnformatted(forthLine.c_str());

            if (!rec.attributes.empty()) {
                std::string attributes;
                U64 i = 0;
                for (const auto& [key, value] : rec.attributes) {
                    attributes += jst::fmt::format("{}{}: {}{}", i == 0 ? "" : "             ",
                                                             key,
                                                             value,
                                                             i == rec.attributes.size() - 1 ? "" : ", \n");
                    i++;
                }
                ImGui::TextFormatted("[ATTRIBUTES: {}]", attributes);
            }

            ImGui::EndTooltip();
        }

        // Resize node by dragging interface logic.
        I32 nodeId;
        if (ImNodes::IsNodeHovered(&nodeId)) {
            auto& node = graph.nodeStates.at(graph.nodeLocaleMap.at(nodeId)).block;

            const auto nodeDims = ImNodes::GetNodeDimensions(nodeId);
            const auto nodeOrigin = ImNodes::GetNodeScreenSpacePos(nodeId);

            F32 nodeWidth = node->state.nodeWidth * scalingFactor;

            bool isNearRightEdge =
                std::abs((nodeOrigin.x + nodeDims.x) - ImGui::GetMousePos().x) < 10.0f &&
                ImGui::GetMousePos().y >= nodeOrigin.y &&
                ImGui::GetMousePos().y <= (nodeOrigin.y + nodeDims.y);

            if (isNearRightEdge && ImGui::IsMouseDown(0) && !state.nodeDragId) {
                ImNodes::SetNodeDraggable(nodeId, false);
                state.nodeDragId = nodeId;
            }

            if (state.nodeDragId) {
                nodeWidth = (ImGui::GetMousePos().x - nodeOrigin.x) - (nodeStyle.NodePadding.x * 2.0f);
            }

            if (isNearRightEdge || state.nodeDragId) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            }

            node->state.nodeWidth = nodeWidth / scalingFactor;
        }

        if (ImGui::IsMouseReleased(0)) {
            if (state.nodeDragId) {
                ImNodes::SetNodeDraggable(state.nodeDragId, true);
                state.nodeDragId = 0;
            }
        }

        ImGui::End();
    }();

    // Update the internal state when a link is deleted.
    I32 linkId;
    if (ImNodes::IsLinkDestroyed(&linkId)) {
        mailboxes.unlinkMailbox = graph.linkLocaleMap.at(linkId);
    }

    // Update the internal state when a link is created.
    I32 startId, endId;
    if (ImNodes::IsLinkCreated(&startId, &endId)) {
        mailboxes.linkMailbox = {graph.pinLocaleMap.at(endId), graph.pinLocaleMap.at(startId)};
    }

    // Draw right-click menu for node actions.
    if (ImNodes::IsNodeHovered(&state.nodeContextMenuNodeId) &&
        (ImGui::IsMouseClicked(ImGuiMouseButton_Right))) {
        ImGui::CloseCurrentPopup();
        ImGui::OpenPopup("##node_context_menu");
    }

    // Draw node context menu.
    if (!state.fullscreenEnabled && ImGui::BeginPopup("##node_context_menu")) {
        const auto& locale = graph.nodeLocaleMap.at(state.nodeContextMenuNodeId);
        const auto& state = graph.nodeStates.at(locale.block());
        const auto moduleEntry = Store::BlockMetadataList().at(state.fingerprint.id);

        // Delete node.
        if (ImGui::MenuItem("Delete Node")) {
            mailboxes.deleteBlockMailbox = locale;
        }

        // Rename node.
        if (ImGui::MenuItem("Rename Node")) {
            this->state.globalModalToggle = true;
            this->state.renameBlockLocale = locale;
            this->state.renameBlockNewId = locale.blockId;
            this->state.globalModalContentId = 6;
        }

        // Enable/disable node toggle.
        if (ImGui::MenuItem("Enable Node", nullptr, state.block->complete())) {
            mailboxes.toggleBlockMailbox = {locale, !state.block->complete()};
        }

        // Reload node.
        if (ImGui::MenuItem("Reload Node")) {
            mailboxes.reloadBlockMailbox = locale;
        }

        // Device backend options.
        if (ImGui::BeginMenu("Backend Device")) {
            for (const auto& [device, _] : moduleEntry.options) {
                const auto enabled = (state.block->device() == device);
                if (ImGui::MenuItem(GetDevicePrettyName(device), nullptr, enabled)) {
                    mailboxes.changeBlockBackendMailbox = {locale, device};
                }
            }
            ImGui::EndMenu();
        }

        // Data type options.
        if (ImGui::BeginMenu("Data Type")) {
            for (const auto& types : moduleEntry.options.at(state.block->device())) {
                const auto& [inputDataType, outputDataType] = types;
                const auto enabled = state.fingerprint.inputDataType == inputDataType &&
                                     state.fingerprint.outputDataType == outputDataType;
                const auto label = (outputDataType.empty()) ? jst::fmt::format("{}", inputDataType) :
                                                              jst::fmt::format("{} -> {}", inputDataType, outputDataType);
                if (ImGui::MenuItem(label.c_str(), NULL, enabled)) {
                    mailboxes.changeBlockDataTypeMailbox = {locale, types};
                }
            }
            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    } else if ((ImGui::IsMouseClicked(0) ||
                ImGui::IsMouseClicked(1) ||
                ImGui::IsMouseClicked(2))) {
        ImGui::CloseCurrentPopup();
        state.nodeContextMenuNodeId = 0;
    }


    return Result::SUCCESS;
}

void Compositor::lock() {
    state.interfaceHalt.wait(true);
    state.interfaceHalt.test_and_set();
}

void Compositor::unlock() {
    state.interfaceHalt.clear();
    state.interfaceHalt.notify_one();
}

}  // namespace Jetstream
