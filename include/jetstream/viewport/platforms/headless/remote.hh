#ifndef JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_REMOTE_HH
#define JETSTREAM_VIEWPORT_PLATFORM_HEADLESS_REMOTE_HH

#include <vector>

#include "jetstream/viewport/adapters/generic.hh"

namespace Jetstream::Viewport {

class Remote {
 public:
    Remote();
    ~Remote();

    Result create(const Viewport::Config& config, const Device& viewport_device);
    Result destroy();

    const Device& inputMemoryDevice() const;

    Result pushNewFrame(const void* data);

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

namespace RemoteHelpers {
std::vector<std::string> PendingSessions(const std::vector<std::string>& waitlist,
                                         const std::vector<std::string>& activeSessions);
}

}  // namespace Jetstream::Viewport

#endif
