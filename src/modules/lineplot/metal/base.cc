#include <algorithm>

#include "../generic.cc"

namespace Jetstream {

namespace {
constexpr NSUInteger kLineplotThreadgroupWidth = 256;
}

// Performance Notes:
// - Uses threadgroup (shared) memory to stage averages and reduce global memory traffic
// - Coalesced memory access pattern for input reads (stride-based indexing)
// - Minimal threadgroup barriers (only 2 per kernel invocation)
// - Loop unrolling not beneficial here as batchSize varies at runtime
// - Further optimization would require profiling to identify actual bottlenecks

static const char shadersSrc[] = R"""(
    #include <metal_stdlib>

    using namespace metal;

    struct Constants {
        ushort batchSize;
        ushort gridSize;
        float normalizationFactor;
        size_t average;
        size_t decimation;
    };

    constant ushort kLineplotThreadgroupWidth = 256;

    kernel void lineplot(constant Constants& constants [[ buffer(0) ]],
                         constant const float *input [[ buffer(1) ]],
                         device float2 *bins [[ buffer(2) ]],
                         uint id[[ thread_position_in_grid ]],
                         ushort thread_idx [[ thread_position_in_threadgroup ]]) {
        threadgroup float sharedX[kLineplotThreadgroupWidth];
        threadgroup float sharedAverage[kLineplotThreadgroupWidth];

        if (id >= constants.gridSize) {
            return;
        }

        // Stage historical averages inside fast shared memory so multiple updates
        // in the same threadgroup do not thrash device memory.
        sharedAverage[thread_idx] = bins[id].y;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint baseIndex = id * constants.decimation;
        const uint stride = constants.gridSize;

        // Compute average amplitude within a batch.
        // Note: Loop unrolling doesn't help here as batchSize is runtime-variable
        float amplitude = 0.0f;
        for (uint i = 0; i < constants.batchSize; ++i) {
            amplitude += input[baseIndex + (i * stride)];
        }
        amplitude = (amplitude * constants.normalizationFactor) - 1.0f;

        // Calculate moving average using the staged value.
        float average = sharedAverage[thread_idx];
        average -= average / constants.average;
        average += amplitude / constants.average;

        // Store result back through threadgroup memory to keep the number of
        // global transactions to a minimum.
        sharedX[thread_idx] = id * 2.0f / (constants.gridSize - 1) - 1.0f;
        sharedAverage[thread_idx] = average;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        bins[id].x = sharedX[thread_idx];
        bins[id].y = sharedAverage[thread_idx];
    }
)""";

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    struct Constants {
        U16 batchSize;
        U16 gridSize;
        F32 normalizationFactor;
        U64 average;
        U64 decimation;
    };

    MTL::ComputePipelineState* lineplotState;
    Tensor<Device::Metal, U8> constants;
};

template<Device D, typename T>
Lineplot<D, T>::Lineplot() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Lineplot<D, T>::~Lineplot() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply compute core using Metal backend.");

    // Compile shaders.

    JST_CHECK(Metal::CompileKernel(shadersSrc, "lineplot", &pimpl->lineplotState));

    // Create constants buffer.

    Metal::CreateConstants<typename Impl::Constants>(*pimpl);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context& ctx) {
    auto* constants = Metal::Constants<typename Impl::Constants>(*pimpl);
    constants->batchSize = gimpl->numberOfBatches;
    constants->gridSize = gimpl->numberOfElements;
    constants->normalizationFactor = gimpl->normalizationFactor;
    constants->average = config.averaging;
    constants->decimation = config.decimation;

    {
        auto cmdEncoder = ctx.metal->commandBuffer()->computeCommandEncoder();
        cmdEncoder->setComputePipelineState(pimpl->lineplotState);
        cmdEncoder->setBuffer(pimpl->constants.data(), 0, 0);
        cmdEncoder->setBuffer(input.buffer.data(), 0, 1);
        cmdEncoder->setBuffer(gimpl->signalPoints.data(), 0, 2);
        const NSUInteger maxThreads = pimpl->lineplotState->maxTotalThreadsPerThreadgroup();
        const NSUInteger threadsPerGroup = std::max<NSUInteger>(
            1,
            std::min<NSUInteger>({maxThreads,
                                  static_cast<NSUInteger>(gimpl->numberOfElements),
                                  kLineplotThreadgroupWidth}));
        cmdEncoder->dispatchThreads(MTL::Size(gimpl->numberOfElements, 1, 1),
                                    MTL::Size(threadsPerGroup, 1, 1));
        cmdEncoder->endEncoding();
    }

    gimpl->updateSignalPointsFlag = true;

    return Result::SUCCESS;
}

JST_LINEPLOT_METAL(JST_INSTANTIATION)
JST_LINEPLOT_METAL(JST_BENCHMARK)

}  // namespace Jetstream
