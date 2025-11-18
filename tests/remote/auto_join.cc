#include <catch2/catch_test_macros.hpp>

#include "jetstream/viewport/platforms/headless/remote.hh"

using Jetstream::Viewport::RemoteHelpers::PendingSessions;

TEST_CASE("PendingSessions filters only new sessions", "[remote][auto-join]") {
    std::vector<std::string> waitlist = {"alpha", "beta", "beta", "gamma", "alpha"};
    std::vector<std::string> sessions = {"beta"};

    const auto pending = PendingSessions(waitlist, sessions);

    REQUIRE(pending == std::vector<std::string>{"alpha", "gamma"});
}

TEST_CASE("PendingSessions handles empty queues", "[remote][auto-join]") {
    REQUIRE(PendingSessions({}, {}) == std::vector<std::string>{});

    std::vector<std::string> waitlist = {"one"};
    std::vector<std::string> sessions = {"one"};
    REQUIRE(PendingSessions(waitlist, sessions).empty());
}
