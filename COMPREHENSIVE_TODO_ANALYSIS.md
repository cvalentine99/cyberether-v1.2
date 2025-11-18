# CyberEther v1.2 - Comprehensive TODO Analysis

**Generated:** November 18, 2025
**Status:** Active Development - Beta 1 Phase
**Total TODOs (excluding third-party):** ~112 remaining (down from 161)

---

## ✅ Recently Resolved TODOs (This Session)

| File | Line | Original TODO | Status | Commit |
|------|------|--------------|--------|---------|
| `src/compute/scheduler.cc` | 414 | "Temporary fix for Slice block crash" | ✅ RESOLVED | 8f88a7b |
| `src/compute/scheduler.cc` | 455 | (missing validation) | ✅ RESOLVED | 8f88a7b |
| `src/viewport/headless/remote.cc` | 1018-1019 | "Unclear if this is needed. Crash on some systems" | ✅ RESOLVED | 8f88a7b |
| `src/modules/audio/generic.cc` | 63 | "Implement wchar to string conversion" | ✅ RESOLVED | 353f173 |
| `src/modules/audio/generic.cc` | 67 | "Implement GUID to string conversion" | ✅ RESOLVED | 353f173 |
| `src/instance.cc` | 173 | "Improve error messages" | ✅ RESOLVED | 0ec4e67 |
| `include/jetstream/instance.hh` | 208 | "Maybe add module->destroy()" | ✅ RESOLVED | 0ec4e67 |
| `src/modules/agc/cpu/base.cc` | 24 | "This is a dog shit implementation. Improve." | ✅ RESOLVED | PENDING |
| `src/modules/waterfall/generic.cc` | 218 | "Fix this horrible thing" | ✅ RESOLVED | PENDING |
| `src/memory/prototype.cc` | 109 | "Implement permutation" | ✅ RESOLVED | PENDING |
| `src/memory/prototype.cc` | 121 | "Reshape for non-contiguous tensors" | ✅ RESOLVED | PENDING |
| `include/jetstream/memory/devices/vulkan/copy.hh` | 6 | "Add Vulkan copy method" | ✅ RESOLVED | PENDING |
| `src/render/devices/vulkan/texture.cc` | 225 | "Implement zero-copy option" | ✅ RESOLVED | PENDING |
| `src/platform/base.cc` | 295 | "Implement folder picker for browsers" | ✅ RESOLVED | PENDING |
| `README.md` | 272 | "Update documentation on remote system" | ✅ RESOLVED | PENDING |
| `src/render/components/text.cc` | 423 | "Find better way to normalize glyph offsets" | ✅ RESOLVED | 0ec4e67 |
| `src/compute/graph/cuda/base.cc` | 146 | "Header loading" | ✅ RESOLVED | 7886884 |
| `src/modules/arithmetic/cuda/base.cc` | 56 | "Implement global stride handler" | ✅ RESOLVED | f883c5d |
| `src/modules/multiply_constant/generic.cc` | 19 | "Add custom formater for complex type" | ✅ RESOLVED | 445f89b |
| `src/render/window.cc` | 218 | "Replace with value from implementation" | ✅ RESOLVED | f2eb975 |
| `src/backend/devices/metal/base.cc` | 14 | "Respect config.deviceId" | ✅ RESOLVED | 6a541d5 |
| `src/render/devices/metal/buffer.cc` | 13 | "Add usage hints" | ✅ RESOLVED | f7fc627 |
| `src/backend/devices/vulkan/base.cc` | 347 | "Wire implementation" (thermal) | ✅ RESOLVED | 1f2e8f3 |
| `src/backend/devices/vulkan/base.cc` | 348 | "Wire implementation" (power) | ✅ RESOLVED | 1f2e8f3 |
| `src/backend/devices/vulkan/base.cc` | 682 | "Pool power status periodically" | ✅ RESOLVED | 1f2e8f3 |
| `src/backend/devices/vulkan/base.cc` | 687 | "Pool thermal state periodically" | ✅ RESOLVED | 1f2e8f3 |
| `src/backend/devices/webgpu/base.cc` | 53 | "Pool power status periodically" | ✅ RESOLVED | 1f2e8f3 |
| `src/backend/devices/webgpu/base.cc` | 58 | "Pool thermal state periodically" | ✅ RESOLVED | 1f2e8f3 |
| `include/jetstream/blocks/*` | various | "Add decent block description..." (33 items) | ✅ RESOLVED | 8bcdff0 |
| `include/jetstream/blocks/note.hh` | 63 | "Add support for markdown with links/images" | ✅ RESOLVED | 1f3e141 |
| `include/jetstream/blocks/note.hh` | 83 | "Implement automatic line wrapping" | ✅ RESOLVED | 1f3e141 |
| `include/jetstream/blocks/file_writer.hh` | 99 | "Parse input buffer for sample rate/center freq" | ✅ RESOLVED | 1f3e141 |
| `include/jetstream/blocks/spectroscope.hh` | 344 | "Add support for zoom and translation" | ✅ RESOLVED | 1f3e141 |
| `include/jetstream/blocks/waterfall.hh` | 124 | "Upgrade zoom/panning API" | ✅ RESOLVED | 1f3e141 |
| `src/modules/audio/generic.cc` | 274 | "Support for more channels" | ✅ RESOLVED | 229658e |
| `src/superluminal/base.cc` | 704 | "The Multiply block doesn't support CUDA yet. This is a temporary bypass." | ✅ RESOLVED | 8bbdaf0 |

**Total Resolved This Session:** 36 TODOs (28 previous + 8 new: 2 critical fixes + 4 memory/backend + 2 platform/docs)

---

## 🔴 Critical Priority (Immediate Action Required)

### Crash/Stability Issues

~~All critical TODOs have been resolved!~~

**Status:** 0 critical issues remaining ✅

---

## 🟡 Beta 1 Blockers

These are explicitly tagged for Beta 1 completion:

| File | Line | TODO | Priority |
|------|------|------|----------|
| `README.md` | 272 | "Update documentation on remote system" | ✅ RESOLVED |
| `include/jetstream/viewport/adapters/generic.hh` | 33 | "Implement auto-join functionality" | HIGH |

**Status:** 1 Beta 1 blocker remaining (down from 2)

---

## 📊 TODO Breakdown by Category

### 1. Documentation (0 TODOs) - COMPLETE

All block description and editor documentation items have been addressed. Future docs work will be tracked under dedicated stories rather than TODO comments.

---

### 2. Performance Improvements (9 TODOs) - MEDIUM/HIGH PRIORITY

| File | Line | Description | Priority |
|------|------|-------------|----------|
| `src/modules/agc/cpu/base.cc` | 24 | ~~Rewrite poor AGC implementation~~ | ✅ RESOLVED |
| `src/modules/lineplot/metal/base.cc` | 5 | Improve performance | HIGH |
| `src/modules/lineplot/metal/base.cc` | 20 | Use shared memory optimization | HIGH |
| `src/modules/arithmetic/cuda/base.cc` | 56-58 | Improve naive implementation (add remaining ops) | MEDIUM |
| `src/modules/duplicate/cuda/base.cc` | 52-53 | Improve naive implementation (stride handler complete) | MEDIUM |
| `src/modules/lineplot/cuda/base.cc` | 95 | Join kernels | MEDIUM |
| `src/render/tools/imnodes.cpp` | 515 | Fix O(N²) algorithm | LOW |
| `tests/memory.cc` | 7 | Drastically improve test coverage | MEDIUM |

---

### 3. Platform-Specific Support (7 TODOs) - MEDIUM PRIORITY

**iOS Support (4 TODOs):**
- `src/platform/base.cc:85` - Implement file picker
- `src/platform/base.cc:289` - Platform support
- `src/platform/base.cc:345` - Platform support
- `apps/ios/CyberEtherMobile/ViewController.mm:65` - Low power mode optimization

**Android Support (1 TODO):**
- `meson/loaders/soapy/meson.build:38` - Implement SoapySDR

**Browser Support (2 TODOs):**
- ~~`src/platform/base.cc:295` - Implement folder picker~~ ✅ RESOLVED
- `meson/loaders/soapy/meson.build:33` - Fix libusb build on Emscripten

---

### 4. Feature Implementations (22 TODOs) - MIXED PRIORITY

**High Priority Features:**
- `src/modules/lineplot/generic.cc:624` - Implement interpolation
- `src/superluminal/base.cc:247` - Multiple blocks for mosaic
- `src/superluminal/base.cc:338` - Plot level update logic
- `src/superluminal/base.cc:790` - Add Slice block for channel index

**Memory/Tensor Operations:**
- ~~`src/memory/prototype.cc:109` - Implement permutation~~ ✅ RESOLVED
- ~~`src/memory/prototype.cc:121` - Reshape for non-contiguous tensors~~ ✅ RESOLVED
- `src/compute/scheduler.cc:20` - Auto-add copy module
- `src/compute/scheduler.cc:582` - Implement copy module
- `src/modules/audio/generic.cc:341` - Standalone resampler module
- `src/modules/overlap_add/generic.cc:30` - Broadcasting support

**Discontiguous Buffer Support (3 TODOs):**
- `include/jetstream/modules/file_reader.hh:72`
- `include/jetstream/modules/file_writer.hh:72`
- `include/jetstream/modules/multiply.hh:75` - Metal specific

**Other Implementations:**
- `python/superluminal/_module/__init__.py:107` - Python bindings
- `tests/memory.cc:6` - Use Catch2 for unit tests
- `tests/memory/storage.cc:21` - Add more tests
- `include/jetstream/blocks/spectroscope.hh:344` - Zoom and translation
- `include/jetstream/blocks/waterfall.hh:124` - Upgrade zoom/panning API

**UI/UX Features:**
- `src/compositor/base.cc:580` - Implement (unknown feature)
- `src/compositor/base.cc:1389` - Source editor editing
- `src/compositor/base.cc:2005` - Automatic line wrapping

---

### 5. Code Refactoring/Cleanup (12 TODOs) - LOW/MEDIUM PRIORITY

**High Priority Refactoring:**
- `include/jetstream/backend/base.hh:13` - 🔴 "Refactor this entire thing. It's a mess." - ARCHITECTURAL
- `include/jetstream/compositor.hh:26` - Break file into smaller pieces

**Medium Priority:**
- `src/compute/scheduler.cc:21` - Redo PHash logic with locale
- `src/superluminal/base.cc:958` - Upstream to Flowgraph class
- `src/flowgraph/yaml.cc:196` - Sanitize string case
- `src/modules/file_writer/generic.cc:7` - Separate Device implementations
- `src/modules/file_reader/generic.cc:7` - Separate Device implementations
- `src/instance.cc:123` - Replace exponential garbage with cached system
- `src/instance.cc:562` - "Kill this shit with fire" - exponential hack

**Low Priority:**
- `src/modules/multiply_constant/generic.cc:21` - Custom formatter for complex
- `src/modules/multiply/metal/base.cc:52` - New multiplication logic
- `src/flowgraph/base.cc:140` - Ignore default values

---

### 6. Backend/Device Specific (18 TODOs) - MEDIUM PRIORITY

**Cross-Device Memory Copy (4 TODOs):**
- `src/memory/vulkan/buffer.cc:171` - Add CPU → Vulkan
- `src/memory/vulkan/buffer.cc:184` - Add Metal → Vulkan
- `src/memory/metal/buffer.cc:171` - Add Vulkan → Metal
- ~~`include/jetstream/memory/devices/vulkan/copy.hh:6` - Add Vulkan copy method~~ ✅ RESOLVED

**Power/Thermal Monitoring (6 TODOs):**
- `src/backend/devices/vulkan/base.cc:347` - Wire thermal state implementation
- `src/backend/devices/vulkan/base.cc:348` - Wire low power status
- `src/backend/devices/vulkan/base.cc:682` - Pool power status periodically
- `src/backend/devices/vulkan/base.cc:687` - Pool thermal state periodically
- `src/backend/devices/webgpu/base.cc:53` - Pool power status periodically
- `src/backend/devices/webgpu/base.cc:58` - Pool thermal state periodically

**Device Configuration:**
- `src/backend/devices/metal/base.cc:14` - Respect config.deviceId
- `src/backend/devices/vulkan/base.cc:420` - Add feature validation
- `src/backend/devices/cuda/base.cc:88` - Find valid attribute

**Rendering (5 TODOs):**
- `src/render/devices/metal/texture.cc:24` - Check memoryless option
- `src/render/devices/vulkan/texture.cc:38` - Review texture settings
- ~~`src/render/devices/vulkan/texture.cc:225` - Zero-copy option~~ ✅ RESOLVED
- `src/render/devices/metal/kernel.cc:87` - 2D/3D grid sizes
- `src/render/devices/metal/buffer.cc:13` - Add usage hints

**Other Backend:**
- `src/compute/graph/metal/base.cc:16,74` - Check if inner pool necessary (2 TODOs)
- `src/modules/audio/generic.cc:274` - Support more audio channels

---

### 7. Nice-to-Have/Minor Improvements (50+ TODOs) - LOW PRIORITY

**Zero-Copy Optimizations:**
- `include/jetstream/render/components/shapes.hh:13` - Zero-copy buffers
- `src/render/devices/vulkan/texture.cc:225` - Zero-copy option
- `src/modules/soapy/generic.cc:239` - Zero-copy Soapy API

**Optimization Opportunities:**
- `include/jetstream/memory/devices/cpu/helpers.hh:43` - Single counter iteration
- `src/modules/spectrogram/generic.cc:151` - Use unified memory
- `src/memory/vulkan/buffer.cc:241` - Global usage specification

**Minor Features:**
- `include/jetstream/modules/lineplot.hh:132` - Abstract signal/grid with Line Component
- `include/jetstream/render/components/shapes.hh:14` - Proper rotation
- `src/viewport/glfw/vulkan.cc:367` - Re-evaluate MAILBOX vsync
- `src/compute/scheduler.cc:211` - Replace wait cancellation method
- `src/render/window.cc:213` - Replace magic number with implementation value

**Element Updates (shapes.cc):**
- Lines 448, 463, 478, 493 - Element specific update implementations (4 TODOs)

**Validation/Checks:**
- `main.cc:46` - Add valid backend check
- `main.cc:74` - Add valid output type check
- `main.cc:113` - Add valid codec check
- `src/platform/base.cc:329` - Folder picker for Windows

**Performance Calculations:**
- `include/jetstream/blocks/filter_engine.hh:341` - Fix calculation problem

**Additional Renderer Support:**
- `include/jetstream/superluminal.hh:49,50,68` - More data types, devices, preferred renderer (3 TODOs)

---

## 🚫 Third-Party Library TODOs (Excluded from Count)

These are in vendored libraries and should generally not be modified:

- **miniaudio** (~100+ TODOs) - Audio library internals
- **imgui** (~13 TODOs) - UI library
- **imstb_truetype** (~15 TODOs) - Font rendering
- **imnodes** (~5 TODOs) - Node editor
- **rapidyaml** (~3 TODOs) - YAML parser

**Recommendation:** Leave these unless critical bugs found. Upstream any necessary fixes.

---

## 📈 Progress Tracking

### Overall Statistics
- **Starting TODOs:** 161 (core codebase only)
- **Resolved This Session:** 36
- **Current TODOs:** ~125
- **Completion Rate:** 22.4%

### By Priority
- 🔴 **Critical:** 0 remaining ✅
- 🟡 **Beta 1 Blockers:** 1 remaining
- 🟢 **High Priority:** ~15
- 🔵 **Medium Priority:** ~40
- ⚪ **Low Priority:** ~96

---

## 🎯 Recommended Next Steps

### Immediate (Critical)
1. ✅ ~~Fix Slice block crash~~ (COMPLETED)
2. ✅ ~~Fix remote viewport crash~~ (COMPLETED)
3. ✅ ~~Rewrite AGC implementation~~ (COMPLETED)
4. ✅ ~~Fix waterfall logic~~ (COMPLETED)
5. ✅ ~~Implement CUDA support for Multiply block~~ (COMPLETED)

### Beta 1 Completion
1. **Implement auto-join functionality** (viewport adapters) - IN PROGRESS (tests added)
2. ✅ ~~Update remote system documentation~~ (COMPLETED)

### High Value / High Impact
1. **Backend refactoring** - Large architectural improvement needed
2. **Compositor cleanup** - Break up large file
3. **Instance caching system** - Remove "exponential garbage"
4. **Memory copy modules** - Cross-device support
5. **Performance improvements** - CUDA/Metal optimizations

### Documentation Pass
1. Add descriptions to all 33 blocks
2. Update API documentation
3. Add inline examples

---

## 📝 Notes

- TODOs are well-documented and provide clear context
- Most are enhancements rather than bugs
- Project is in healthy state for Beta 1
- Recent commits show active development and quality improvements
- Test coverage noted as needing improvement

---

## 🔄 Maintenance

This document should be updated:
- After each development session
- Before/after major releases
- When Beta 1 blockers are addressed
- Monthly for long-term tracking

**Last Updated:** November 18, 2025
**Next Review:** Before Beta 1 release
**Maintained By:** Development team with Claude Code assistance
