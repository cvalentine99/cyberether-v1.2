#ifndef JETSTREAM_BLOCK_AGC_BASE_HH
#define JETSTREAM_BLOCK_AGC_BASE_HH

#include "jetstream/block.hh"
#include "jetstream/instance.hh"
#include "jetstream/modules/agc.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class AGC : public Block {
 public:
    // Configuration

    struct Config {
        JST_SERDES();
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "agc";
    }

    std::string name() const {
        return "AGC";
    }

    std::string summary() const {
        return "Stabilizes signal amplitude.";
    }

    std::string description() const {
        return "Automatic Gain Control (AGC) dynamically adjusts the gain of the input signal "
               "to maintain a consistent output amplitude. Takes a complex or real-valued tensor "
               "as input and produces an output tensor of the same type with normalized amplitude. "
               "Useful for stabilizing signal levels in SDR applications and audio processing.";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            agc, "agc", {}, {
                .buffer = input.buffer,
            },
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, agc->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        if (agc) {
            JST_CHECK(instance().eraseModule(agc->locale()));
        }

        return Result::SUCCESS;
    }

 private:
    std::shared_ptr<Jetstream::AGC<D, IT>> agc;

    JST_DEFINE_IO()
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(AGC, is_specialized<Jetstream::AGC<D, IT>>::value &&
                      std::is_same<OT, void>::value)

#endif
