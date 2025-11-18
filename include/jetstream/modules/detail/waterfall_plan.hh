#ifndef JETSTREAM_MODULES_DETAIL_WATERFALL_PLAN_HH
#define JETSTREAM_MODULES_DETAIL_WATERFALL_PLAN_HH

#include <vector>

namespace Jetstream::detail {

struct WaterfallUpdateRange {
    int startRow = 0;
    int blockCount = 0;

    [[nodiscard]] constexpr bool empty() const {
        return blockCount <= 0;
    }
};

inline std::vector<WaterfallUpdateRange> ComputeWaterfallUpdateRanges(int last, int current, int height) {
    std::vector<WaterfallUpdateRange> plan;
    if (height <= 0) {
        return plan;
    }

    auto normalize = [height](int value) {
        if (value < 0) {
            return 0;
        }
        if (value > height) {
            return height;
        }
        return value;
    };

    last = normalize(last);
    current = normalize(current);

    if (current == last) {
        return plan;
    }

    if (current > last) {
        plan.push_back({last, current - last});
        return plan;
    }

    if (last < height) {
        plan.push_back({last, height - last});
    }

    if (current > 0) {
        plan.push_back({0, current});
    }

    return plan;
}

}  // namespace Jetstream::detail

#endif
