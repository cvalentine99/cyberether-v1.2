#include <thread>
#include <vector>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "jetstream/memory/base.hh"

using namespace Jetstream;

TEST_CASE("Tensor locale metadata propagates across copies", "[memory][locale]") {
    Tensor<Device::CPU, F32> tensor;
    tensor.set_locale({"blockA", "moduleB", "pinC"});

    REQUIRE(tensor.locale().blockId == "blockA");
    REQUIRE(tensor.locale().moduleId == "moduleB");
    REQUIRE(tensor.locale().pinId == "pinC");

    Tensor<Device::CPU, F32> copy(tensor);
    REQUIRE(copy.locale().blockId == "blockA");
    REQUIRE(copy.locale().moduleId == "moduleB");
    REQUIRE(copy.locale().pinId == "pinC");

    Tensor<Device::CPU, F32> assigned;
    assigned = tensor;
    REQUIRE(assigned.locale().blockId == "blockA");
    REQUIRE(assigned.locale().moduleId == "moduleB");
    REQUIRE(assigned.locale().pinId == "pinC");
}

TEST_CASE("CPU tensor allocation maintains size and references", "[memory][allocation]") {
    Tensor<Device::CPU, U64> tensor({1, 2, 3, 4});

    REQUIRE(tensor.size() == 24);
    REQUIRE(tensor.references() == 1);
    REQUIRE(tensor.data() != nullptr);
}

TEST_CASE("CPU tensor copy and move semantics update references", "[memory][copy]") {
    Tensor<Device::CPU, U64> tensor({24});

    auto copy = tensor;
    REQUIRE(tensor.references() == 2);
    REQUIRE(copy.references() == 2);

    Tensor<Device::CPU, U64> moved(std::move(copy));
    REQUIRE(moved.references() == 1);
    REQUIRE(moved.data() != nullptr);
}

TEST_CASE("CPU memory copy replicates contents", "[memory][copy]") {
    Tensor<Device::CPU, U64> src({16});
    for (U64 i = 0; i < src.size(); ++i) {
        src[i] = i * 3;
    }

    Tensor<Device::CPU, U64> dst(src.shape());
    REQUIRE(Memory::Copy(dst, src) == Result::SUCCESS);

    for (U64 i = 0; i < src.size(); ++i) {
        REQUIRE(dst[i] == src[i]);
    }
}

TEST_CASE("Tensor reference counting stays synchronized across threads", "[memory][sync]") {
    Tensor<Device::CPU, U64> shared({32});
    for (U64 i = 0; i < shared.size(); ++i) {
        shared[i] = i;
    }

    constexpr int kWorkers = 4;
    std::vector<std::thread> workers;
    std::vector<bool> results(kWorkers, false);

    for (int i = 0; i < kWorkers; ++i) {
        workers.emplace_back([&, idx = i]() {
            Tensor<Device::CPU, U64> local(shared);
            Tensor<Device::CPU, U64> scratch(local.shape());
            if (Memory::Copy(scratch, local) != Result::SUCCESS) {
                return;
            }

            bool matches = true;
            for (U64 j = 0; j < scratch.size(); ++j) {
                if (scratch[j] != local[j]) {
                    matches = false;
                    break;
                }
            }
            results[idx] = matches;
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    REQUIRE(shared.references() == 1);
    for (bool ok : results) {
        REQUIRE(ok);
    }
}

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(4);
    return Catch::Session().run(argc, argv);
}
