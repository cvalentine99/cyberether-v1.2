#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct AGC<D, T>::Impl {
    F32 envelope = 0.0f;
    F32 gain = 1.0f;
};

template<Device D, typename T>
AGC<D, T>::AGC() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
AGC<D, T>::~AGC() {
    impl.reset();
}

template<typename SampleType>
inline F32 ComputeMagnitude(const SampleType& sample) {
    if constexpr (std::is_same_v<SampleType, CF32>) {
        return std::abs(sample);
    } else {
        return std::abs(sample);
    }
}

inline F32 TimeConstantToCoefficient(const F32 timeConstant, const F32 sampleRate) {
    if (timeConstant <= 0.0f || sampleRate <= 0.0f) {
        return 0.0f;
    }

    const F32 samples = timeConstant * sampleRate;
    if (samples <= 1.0f) {
        return 0.0f;
    }

    return std::exp(-1.0f / samples);
}

template<Device D, typename T>
Result AGC<D, T>::compute(const Context&) {
    const U64 sampleCount = input.buffer.size();
    if (sampleCount == 0) {
        return Result::SUCCESS;
    }

    const F32 attackCoeff = TimeConstantToCoefficient(config.attackTime, config.sampleRate);
    const F32 releaseCoeff = TimeConstantToCoefficient(config.releaseTime, config.sampleRate);
    const F32 epsilon = std::numeric_limits<F32>::epsilon();
    const F32 maxGainValue = std::max(config.maxGain, 1.0f);
    const F32 minGainValue = 1.0f / maxGainValue;

    for (U64 idx = 0; idx < sampleCount; ++idx) {
        const T& sample = input.buffer[idx];
        const F32 magnitude = ComputeMagnitude(sample);

        const F32 envelopeCoeff = (magnitude > impl->envelope) ? attackCoeff : releaseCoeff;
        impl->envelope = magnitude + envelopeCoeff * (impl->envelope - magnitude);

        F32 desiredGain = config.targetLevel / std::max(impl->envelope, epsilon);
        desiredGain = std::clamp(desiredGain, minGainValue, maxGainValue);

        const F32 gainCoeff = (desiredGain < impl->gain) ? attackCoeff : releaseCoeff;
        impl->gain = desiredGain + gainCoeff * (impl->gain - desiredGain);

        output.buffer[idx] = sample * impl->gain;
    }

    return Result::SUCCESS;
}

JST_AGC_CPU(JST_INSTANTIATION)
JST_AGC_CPU(JST_BENCHMARK)

}  // namespace Jetstream
