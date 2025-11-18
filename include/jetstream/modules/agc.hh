#ifndef JETSTREAM_MODULES_AGC_HH
#define JETSTREAM_MODULES_AGC_HH

#include "jetstream/logger.hh"
#include "jetstream/module.hh"
#include "jetstream/types.hh"
#include "jetstream/memory/base.hh"
#include "jetstream/compute/graph/base.hh"

namespace Jetstream {

#define JST_AGC_CPU(MACRO) \
    MACRO(AGC, CPU, CF32) \
    MACRO(AGC, CPU, F32)

template<Device D, typename T = CF32>
class AGC : public Module, public Compute {
 public:
    AGC();
    ~AGC();

    // Configuration

    struct Config {
        F32 targetLevel = 0.5f;
        F32 maxGain = 20.0f;
        F32 attackTime = 0.01f;
        F32 releaseTime = 0.1f;
        F32 sampleRate = 48000.0f;

        JST_SERDES(targetLevel, maxGain, attackTime, releaseTime, sampleRate);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, T> buffer;

        JST_SERDES_INPUT(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final;

    // Constructor

    Result create();

 protected:
    Result compute(const Context& ctx) final;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    JST_DEFINE_IO()
};

#ifdef JETSTREAM_MODULE_AGC_CPU_AVAILABLE
JST_AGC_CPU(JST_SPECIALIZATION);
#endif

}  // namespace Jetstream

#endif
