#include "jetstream/modules/arithmetic.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename IT>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    // 2D reduction along axis 0
    JST_BENCHMARK_RUN("128x8000 (axis=0)", {
        .axis = 0 COMMA
    }, {
        .buffer = Tensor<D COMMA IT>({128 COMMA 8000}) COMMA
    }, IT);

    // 2D reduction along axis 1
    JST_BENCHMARK_RUN("128x8000 (axis=1)", {
        .axis = 1 COMMA
    }, {
        .buffer = Tensor<D COMMA IT>({128 COMMA 8000}) COMMA
    }, IT);

    // 3D reduction along axis 0
    JST_BENCHMARK_RUN("2x64x8000 (axis=0)", {
        .axis = 0 COMMA
    }, {
        .buffer = Tensor<D COMMA IT>({2 COMMA 64 COMMA 8000}) COMMA
    }, IT);

    // 3D reduction along axis 1
    JST_BENCHMARK_RUN("2x64x8000 (axis=1)", {
        .axis = 1 COMMA
    }, {
        .buffer = Tensor<D COMMA IT>({2 COMMA 64 COMMA 8000}) COMMA
    }, IT);

    // 3D reduction along axis 2
    JST_BENCHMARK_RUN("2x64x8000 (axis=2)", {
        .axis = 2 COMMA
    }, {
        .buffer = Tensor<D COMMA IT>({2 COMMA 64 COMMA 8000}) COMMA
    }, IT);
}

}  // namespace Jetstream