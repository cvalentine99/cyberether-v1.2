#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "jetstream/modules/agc.hh"

using namespace Jetstream;

namespace {

template<typename SampleType>
Tensor<Device::CPU, SampleType> RunAgc(const std::vector<SampleType>& input,
        const typename AGC<Device::CPU, SampleType>::Config& config) {
    AGC<Device::CPU, SampleType> agc;

    typename AGC<Device::CPU, SampleType>::Input in;
    in.buffer = Tensor<Device::CPU, SampleType>({static_cast<U64>(input.size())});
    std::copy(input.begin(), input.end(), in.buffer.begin());

    agc.init_benchmark_mode(config, in);
    REQUIRE(agc.create() == Result::SUCCESS);

    Compute::Context ctx{};
    REQUIRE(agc.compute(ctx) == Result::SUCCESS);

    return agc.getOutputBuffer();
}

}  // namespace

TEST_CASE("AGC keeps silence stable", "[agc]") {
    AGC<Device::CPU, F32>::Config config;
    config.targetLevel = 0.75f;
    config.maxGain = 8.0f;
    config.attackTime = 0.0f;
    config.releaseTime = 0.0f;

    std::vector<F32> input(32, 0.0f);
    const auto output = RunAgc(input, config);

    for (const auto sample : output) {
        REQUIRE(sample == Approx(0.0f));
    }
}

TEST_CASE("AGC clamps excessive gain on tiny signals", "[agc]") {
    AGC<Device::CPU, F32>::Config config;
    config.targetLevel = 1.0f;
    config.maxGain = 4.0f;
    config.attackTime = 0.0f;
    config.releaseTime = 0.0f;

    std::vector<F32> input(8, 1e-6f);
    const auto output = RunAgc(input, config);

    const F32 expected = input.front() * config.maxGain;
    for (const auto sample : output) {
        REQUIRE(sample == Approx(expected));
    }
}

TEST_CASE("AGC reduces loud complex samples without instability", "[agc]") {
    AGC<Device::CPU, CF32>::Config config;
    config.targetLevel = 1.0f;
    config.maxGain = 32.0f;
    config.attackTime = 0.0f;
    config.releaseTime = 0.0f;

    const std::vector<CF32> input = {CF32(4.0f, -4.0f)};
    const auto output = RunAgc(input, config);

    REQUIRE(output.size() == 1);
    const auto value = output[0];
    const F32 magnitude = std::abs(value);
    REQUIRE(magnitude == Approx(config.targetLevel));
    REQUIRE(std::isfinite(value.real()));
    REQUIRE(std::isfinite(value.imag()));
}
