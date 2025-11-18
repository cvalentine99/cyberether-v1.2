#ifndef JETSTREAM_COMPOSITOR_TYPES_HH
#define JETSTREAM_COMPOSITOR_TYPES_HH

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/types.hh"

namespace Jetstream::CompositorDetail {

using CreateBlockMail = std::pair<std::string, Device>;
using LinkMail = std::pair<Locale, Locale>;
using UnlinkMail = std::pair<Locale, Locale>;
using DeleteBlockMail = Locale;
using ReloadBlockMail = Locale;
using RenameBlockMail = std::pair<Locale, std::string>;
using ChangeBlockBackendMail = std::pair<Locale, Device>;
using ChangeBlockDataTypeMail = std::pair<Locale, std::tuple<std::string, std::string>>;
using ToggleBlockMail = std::pair<Locale, bool>;

using LinkId = U64;
using PinId = U64;
using NodeId = U64;

struct NodeState {
    std::shared_ptr<Block> block;

    Parser::RecordMap inputMap;
    Parser::RecordMap outputMap;
    Parser::RecordMap stateMap;
    Block::Fingerprint fingerprint;
    std::string title;

    NodeId id;
    U64 clusterLevel;
    std::unordered_map<PinId, Locale> inputs;
    std::unordered_map<PinId, Locale> outputs;
    std::unordered_set<NodeId> edges;
    bool enabled = true;
};

}  // namespace Jetstream::CompositorDetail

#endif
