#include <algorithm>
#include <cassert>

#include "jetstream/memory/prototype.hh"

namespace Jetstream {

void TensorPrototype::initialize(const std::vector<U64>& shape, const U64& element_size) {
    prototype.element_size = element_size;

    if (shape.empty()) {
        return;
    }

    prototype.shape = shape;

    prototype.stride.resize(prototype.shape.size());
    for (U64 i = 0; i < prototype.shape.size(); i++) {
        prototype.stride[i] = 1;
        for (U64 j = i + 1; j < prototype.shape.size(); j++) {
            prototype.stride[i] *= prototype.shape[j];
        }
    }

    assert(prototype.stride.size() == prototype.shape.size());

    prototype.hash = std::rand() + 1;
    prototype.contiguous = true;

    update_cache();
}

void TensorPrototype::update_cache() {
    prototype.backstride.resize(prototype.shape.size());
    prototype.shape_minus_one.resize(prototype.shape.size());

    for (U64 i = 0; i < prototype.shape.size(); i++) {
        prototype.shape_minus_one[i] = prototype.shape[i] - 1;
        prototype.backstride[i] = prototype.stride[i] * prototype.shape_minus_one[i];
    }

    prototype.size = 1;
    for (const auto& dim : prototype.shape) {
        prototype.size *= dim;
    }

    prototype.size_bytes = prototype.size *
                           prototype.element_size;

    prototype.offset_bytes = prototype.offset *
                             prototype.element_size;
}

const U64& TensorPrototype::shape(const U64& idx) const noexcept {
    return prototype.shape[idx];
}

const U64& TensorPrototype::stride(const U64& idx) const noexcept {
    return prototype.stride[idx];
}

U64 TensorPrototype::rank() const noexcept {
    return prototype.shape.size();
}

U64 TensorPrototype::ndims() const noexcept {
    return prototype.shape.size();
}

void TensorPrototype::set_locale(const Locale& locale) noexcept {
    prototype.locale = locale;
}

void TensorPrototype::set_hash(const U64& hash) noexcept {
    prototype.hash = hash;
}

U64 TensorPrototype::shape_to_offset(const std::vector<U64>& shape) const {
    U64 index = prototype.offset;
    U64 pad = shape.size() - prototype.stride.size();
    for (U64 i = 0; i < prototype.stride.size(); i++) {
        index += shape[pad + i] * prototype.stride[i];
    }
    return index;
}

void TensorPrototype::offset_to_shape(U64 index, std::vector<U64>& shape) const {
    index -= prototype.offset;
    for (U64 i = 0; i < prototype.stride.size(); i++) {
        shape[i] = index / prototype.stride[i];
        index -= shape[i] * prototype.stride[i];
    }
}

void TensorPrototype::expand_dims(const U64& axis) {
    prototype.shape.insert(prototype.shape.begin() + axis, 1);
    const U64& stride = (axis == 0) ? prototype.stride[0] : prototype.stride[axis - 1];
    prototype.stride.insert(prototype.stride.begin() + axis, stride);
    update_cache();
}

void TensorPrototype::squeeze_dims(const U64& axis) {
    assert(prototype.shape[axis] == 1);
    prototype.shape.erase(prototype.shape.begin() + axis);
    prototype.stride.erase(prototype.stride.begin() + axis);
    update_cache();
}

Result TensorPrototype::permutation(const std::vector<U64>& permutation) {
    if (permutation.size() != prototype.shape.size()) {
        JST_ERROR("[MEMORY] Permutation size ({}) must match tensor rank ({}).",
                  permutation.size(),
                  prototype.shape.size());
        return Result::ERROR;
    }

    std::vector<bool> seen(permutation.size(), false);
    std::vector<U64> newShape(permutation.size());
    std::vector<U64> newStride(permutation.size());

    for (size_t i = 0; i < permutation.size(); ++i) {
        const auto axis = permutation[i];
        if (axis >= prototype.shape.size()) {
            JST_ERROR("[MEMORY] Permutation axis '{}' is out of bounds for rank '{}'.",
                      axis,
                      prototype.shape.size());
            return Result::ERROR;
        }
        if (seen[axis]) {
            JST_ERROR("[MEMORY] Permutation axis '{}' is repeated.", axis);
            return Result::ERROR;
        }
        seen[axis] = true;
        newShape[i] = prototype.shape[axis];
        newStride[i] = prototype.stride[axis];
    }

    prototype.shape = std::move(newShape);
    prototype.stride = std::move(newStride);
    update_cache();

    return Result::SUCCESS;
}

Result TensorPrototype::reshape(const std::vector<U64>& shape) {
    if (shape.empty()) {
        JST_ERROR("[MEMORY] Cannot reshape to empty shape.");
        return Result::ERROR;
    }

    const U64& og_size = prototype.size;

    U64 new_size = 1;
    for (const auto& dim : shape) {
        if (dim == 0) {
            JST_ERROR("[MEMORY] Cannot reshape to shape with zero dimension.");
            return Result::ERROR;
        }
        new_size *= dim;
    }

    if (og_size != new_size) {
        JST_ERROR("[MEMORY] Cannot reshape from size {} to size {}.", og_size, new_size);
        return Result::ERROR;
    }

    auto assignContiguousStride = [&]() {
        prototype.shape = shape;

        prototype.stride.resize(prototype.shape.size());
        for (U64 i = 0; i < prototype.shape.size(); i++) {
            prototype.stride[i] = 1;
            for (U64 j = i + 1; j < prototype.shape.size(); j++) {
                prototype.stride[i] *= prototype.shape[j];
            }
        }

        prototype.contiguous = true;
        update_cache();

        return Result::SUCCESS;
    };

    if (prototype.contiguous) {
        return assignContiguousStride();
    }

    struct Chunk {
        U64 size = 0;
        U64 innerStride = 0;
        bool broadcast = false;
    };

    std::vector<Chunk> chunks;
    const auto rank = prototype.shape.size();
    if (rank == 0) {
        JST_ERROR("[MEMORY] Cannot reshape empty tensor.");
        return Result::ERROR;
    }

    for (int64_t i = static_cast<int64_t>(rank) - 1; i >= 0;) {
        if (prototype.stride[i] == 0) {
            chunks.push_back({prototype.shape[i], 0, true});
            --i;
            continue;
        }

        U64 size = prototype.shape[i];
        U64 innerStride = prototype.stride[i];
        int64_t head = i;

        while (head > 0 &&
               prototype.stride[head - 1] != 0 &&
               prototype.stride[head - 1] == prototype.shape[head] * prototype.stride[head]) {
            --head;
            size *= prototype.shape[head];
        }

        chunks.push_back({size, innerStride, false});
        i = head - 1;
    }

    std::reverse(chunks.begin(), chunks.end());

    std::vector<U64> newStride(shape.size(), 0);
    size_t remainingDims = shape.size();

    for (int chunkIndex = static_cast<int>(chunks.size()) - 1; chunkIndex >= 0; --chunkIndex) {
        const auto& chunk = chunks[chunkIndex];
        if (remainingDims == 0) {
            JST_ERROR("[MEMORY] Reshape failed: insufficient dimensions to map chunk.");
            return Result::ERROR;
        }

        size_t chunkEnd = remainingDims;
        size_t chunkBegin = chunkEnd - 1;
        U64 filled = shape[chunkBegin];
        while (filled < chunk.size && chunkBegin > 0) {
            --chunkBegin;
            filled *= shape[chunkBegin];
        }

        if (filled != chunk.size) {
            JST_ERROR("[MEMORY] Cannot reshape view with current strides."
                      " Requested shape {} is incompatible with layout {}.",
                      shape,
                      prototype.shape);
            return Result::ERROR;
        }

        if (chunk.broadcast || chunk.innerStride == 0) {
            for (size_t idx = chunkBegin; idx < chunkEnd; ++idx) {
                newStride[idx] = 0;
            }
        } else {
            for (size_t idx = chunkEnd; idx-- > chunkBegin;) {
                if (idx == chunkEnd - 1) {
                    newStride[idx] = chunk.innerStride;
                } else {
                    newStride[idx] = newStride[idx + 1] * shape[idx + 1];
                }
            }
        }

        remainingDims = chunkBegin;
    }

    if (remainingDims != 0) {
        JST_ERROR("[MEMORY] Cannot reshape view; leftover dimensions remain.");
        return Result::ERROR;
    }

    prototype.shape = shape;
    prototype.stride = newStride;
    // Preserve the current contiguity state since this is a view reshape.
    update_cache();

    return Result::SUCCESS;
}

Result TensorPrototype::broadcast_to(const std::vector<U64>& shape) {
    if (shape.size() < prototype.shape.size()) {
        JST_ERROR("[MEMORY] Cannot broadcast shape: {} -> {}.", prototype.shape, shape);
        return Result::ERROR;
    }

    if (shape.size() > prototype.shape.size()) {
        for (U64 i = 0; i < shape.size() - prototype.shape.size(); i++) {
            expand_dims(0);
        }
    }

    bool contiguous = prototype.contiguous;
    std::vector<U64> new_shape(shape.size());
    std::vector<U64> new_stride(shape.size());

    for (U64 i = 0; i < shape.size(); i++) {
        if (prototype.shape[i] != shape[i]) {
            if (prototype.shape[i] == 1) {
                new_shape[i] = shape[i];
                new_stride[i] = 0;
            } else if (shape[i] == 1) {
                new_shape[i] = prototype.shape[i];
                new_stride[i] = prototype.stride[i];
            } else {
                JST_ERROR("[MEMORY] Cannot broadcast shape: {} -> {}.", prototype.shape, shape);
                return Result::ERROR;
            }
        } else {
            new_shape[i] = prototype.shape[i];
            new_stride[i] = prototype.stride[i];
        }
        contiguous &= new_stride[i] != 0;
    }

    JST_TRACE("[MEMORY] Broadcast shape: {} -> {}.", prototype.shape, new_shape);
    JST_TRACE("[MEMORY] Broadcast stride: {} -> {}.", prototype.stride, new_stride);
    JST_TRACE("[MEMORY] Broadcast contiguous: {} -> {}.", prototype.contiguous, contiguous);

    prototype.shape = new_shape;
    prototype.stride = new_stride;
    prototype.contiguous = contiguous;

    update_cache();

    return Result::SUCCESS;
}

Result TensorPrototype::slice(const std::vector<Token>& slice) {
    std::vector<U64> shape;
    std::vector<U64> stride;
    U64 offset = 0;
    U64 dim = 0;
    bool ellipsis_used = false;

    for (const auto& token : slice) {
        switch (token.get_type()) {
            case Token::Type::Number: {
                if (dim >= prototype.shape.size()) {
                    JST_ERROR("[MEMORY] Index exceeds array dimensions.");
                    return Result::ERROR;
                }

                const U64 index = token.get_a();
                if (index >= prototype.shape[dim]) {
                    JST_ERROR("[MEMORY] Index exceeds array dimensions.");
                    return Result::ERROR;
                }

                offset += index * prototype.stride[dim];
                dim++;
                break;
            }
            case Token::Type::Colon: {
                if (dim >= prototype.shape.size()) {
                    JST_ERROR("[MEMORY] Index exceeds array dimensions.");
                    return Result::ERROR;
                }

                const U64 start = token.get_a();
                U64 end = token.get_b();
                const U64 step = token.get_c();

                if (end == 0) {
                    end = prototype.shape[dim];
                }

                if (step == 0) {
                    JST_ERROR("[MEMORY] Slice step cannot be zero.");
                    return Result::ERROR;
                }

                if (start >= prototype.shape[dim] || end > prototype.shape[dim]) {
                    JST_ERROR("[MEMORY] Slice index exceeds array dimensions.");
                    return Result::ERROR;
                }

                if (start >= end) {
                    JST_ERROR("[MEMORY] Slice start index must be less than end index.");
                    return Result::ERROR;
                }

                shape.push_back((end - start + step - 1) / step);
                stride.push_back(prototype.stride[dim] * step);
                offset += start * prototype.stride[dim];
                dim++;
                break;
            }
            case Token::Type::Ellipsis: {
                if (ellipsis_used) {
                    JST_ERROR("[MEMORY] Ellipsis used more than once.");
                    return Result::ERROR;
                }
                ellipsis_used = true;

                const U64 remaining_dims = prototype.shape.size() - (slice.size() - 1);
                while (dim < remaining_dims) {
                    shape.push_back(prototype.shape[dim]);
                    stride.push_back(prototype.stride[dim]);
                    dim++;
                }
                break;
            }
        }
    }

    if (!ellipsis_used) {
        while (dim < prototype.shape.size()) {
            shape.push_back(prototype.shape[dim]);
            stride.push_back(prototype.stride[dim]);
            dim++;
        }
    }

    JST_TRACE("[MEMORY] Slice shape: {} -> {}.", prototype.shape, shape);
    JST_TRACE("[MEMORY] Slice stride: {} -> {}.", prototype.stride, stride);
    JST_TRACE("[MEMORY] Slice offset: {}.", offset);

    prototype.contiguous = true;
    if (!shape.empty()) {
        U64 expected_stride = 1;
        for (int64_t i = shape.size() - 1; i >= 0; i--) {
            if (stride[i] != expected_stride) {
                prototype.contiguous = false;
                break;
            }
            expected_stride *= shape[i];
        }
    }

    prototype.shape = shape;
    prototype.stride = stride;
    prototype.offset = offset;

    update_cache();

    return Result::SUCCESS;
}

}  // namespace Jetstream
