#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "jetstream/modules/detail/waterfall_plan.hh"

using namespace Jetstream::detail;

TEST_CASE("Waterfall update plan handles monotonic regions", "[waterfall]") {
    const auto plan = ComputeWaterfallUpdateRanges(4, 10, 32);
    REQUIRE(plan.size() == 1);
    REQUIRE(plan[0].startRow == 4);
    REQUIRE(plan[0].blockCount == 6);
}

TEST_CASE("Waterfall update plan splits wrapped regions", "[waterfall]") {
    const auto plan = ComputeWaterfallUpdateRanges(28, 2, 32);
    REQUIRE(plan.size() == 2);
    REQUIRE(plan[0].startRow == 28);
    REQUIRE(plan[0].blockCount == 4);
    REQUIRE(plan[1].startRow == 0);
    REQUIRE(plan[1].blockCount == 2);
}

TEST_CASE("Waterfall update plan tolerates no movement", "[waterfall]") {
    const auto plan = ComputeWaterfallUpdateRanges(5, 5, 32);
    REQUIRE(plan.empty());
}

TEST_CASE("Waterfall update plan guards invalid dimensions", "[waterfall]") {
    const auto plan = ComputeWaterfallUpdateRanges(5, 8, 0);
    REQUIRE(plan.empty());
}
