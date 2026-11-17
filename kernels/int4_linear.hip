// Linear layers on int4 weights that stay packed: y = x W^T, read straight from the checkpoint's packed words. The
// weight a column holds is what the checkpoint's own layer unpacks: (nibble - zero) * scale, rounded to BF16. Products
// of BF16 pairs are summed in FP32 and the sums rounded to BF16.
//
// Layout of one layer, as compressed-tensors stores it, with groups of 32 columns:
//   packed [out, in / 8]    one word holds 8 columns of a row, 4 bits each, the first column in the lowest bits
//   scale  [out, in / 32]   BF16, one per group of a row
//   zero   [out / 8, in / 32]   one word holds the zero points of 8 rows, 4 bits each; absent for symmetric weights
//
// Memory is read a whole cache line at a time: 16 packed words are 128 columns. Reading a word here and a word there
// fetches each line many times over.
//
// Two kernels share the work by the number of rows of x, and the second has a form of its own for 48 to 128 rows.
//
// For one row of x, a wave takes one row of the weight and its lanes split the columns, a group of 32 columns to a lane
// at a time, so the wave reads 512 contiguous bytes of the row at once. Each lane multiplies the pairs of weights it
// decodes with the device's BF16 dot product instruction, and the lanes add up their sums at the end.
//
// For 2 to 512 rows of x, the weight is decoded once into shared memory and the rows share it. A workgroup takes one
// or two tiles of 16 rows of the weight and decodes 512 columns of them at a time. Each of its waves takes a tile of
// 16 rows of x and multiplies with the device's 16x16x16 BF16 matrix instruction: it loads 64 columns of its rows
// once and multiplies them with every weight tile of the workgroup.
//
// For 48 to 128 rows the matrix kernel reads ahead of its work. While a workgroup multiplies one unit it already holds
// the packed words of the next one in registers, and each wave reads its x ahead of the products that use it. Its
// barriers wait for shared memory only, so the reads stay in flight across them. A wave takes two or three tiles of x,
// so each weight fragment it reads from shared memory feeds two or three products, and four or eight waves share the
// columns of each unit, which keeps the workgroup's waves up for the same shared memory; their sums are added at the
// end. Its decode goes through FP16 and takes fewer instructions for the same weights.
//
// `w4a16_route` chooses among them, or says why a product is the checkpoint's own layer's to run. In the
// developer mode every product is recorded with the path it took (see instrument.h).

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "api.h"
#include "instrument.h"

namespace {

constexpr int kGroup = 32;        // Columns that share a scale and a zero point.
constexpr int kUnit = 512;        // Columns a workgroup of the matrix kernel decodes at a time.
constexpr int kLines = 4;         // Cache lines of packed words in a unit.
constexpr int kLineWords = 16;    // Packed words in a cache line: 128 columns.
constexpr int kRowWaves = 4;      // Waves in a workgroup of the dot product kernel, a row of the weight each.
constexpr int kTile = 16;         // Rows on each side of a matrix product.
constexpr int kRun = 64;          // Columns a lane loads at once: four products.
constexpr int kProducts = kUnit / kTile;  // Matrix products of one weight tile in a unit.
constexpr int kRows = 512;        // Rows of x one call takes: a workgroup of 32 waves.
constexpr int kTwoTiles = 48;     // From here a workgroup takes two weight tiles, below it one.
constexpr int kAheadLast = 128;   // Up to here, from kTwoTiles, the matrix kernel reads ahead.

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef short bf16x8 __attribute__((ext_vector_type(8)));  // Eight BF16 as the matrix instruction takes them.
typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef bf16x8 bf16x8_unaligned __attribute__((aligned(2)));

__device__ inline bf16x2 as_pair(uint32_t bits) { return __builtin_bit_cast(bf16x2, bits); }

__device__ inline float from_bits(uint32_t bits) { return __builtin_bit_cast(float, bits); }

__device__ inline uint32_t to_bits(float value) { return __builtin_bit_cast(uint32_t, value); }

// The bits of a float rounded to BF16, to nearest and ties to even, in the upper half of the word. The lower half is
// not cleared.
__device__ inline uint32_t rounded(float value) {
    const uint32_t bits = to_bits(value);
    return bits + 0x7FFFu + ((bits >> 16) & 1u);
}

// The weights of two neighbouring columns as a pair of BF16: byte `index` of a packed word, with the zero point and
// the scale of its row and group.
__device__ inline uint32_t weight_pair(uint32_t word, int index, int zero, float scale) {
    const int low = static_cast<int>((word >> (8 * index)) & 15u) - zero;
    const int high = static_cast<int>((word >> (8 * index + 4)) & 15u) - zero;
    return (rounded(static_cast<float>(low) * scale) >> 16) | (rounded(static_cast<float>(high) * scale) & 0xFFFF0000u);
}

// The weights of eight neighbouring columns: one packed word, with the zero point and the scale of its row and group.
__device__ inline bf16x8 weight_word(uint32_t word, int zero, float scale) {
    bf16x8 weights;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int value = static_cast<int>((word >> (4 * i)) & 15u) - zero;
        weights[i] = static_cast<short>(rounded(static_cast<float>(value) * scale) >> 16);
    }
    return weights;
}

// The same eight weights through FP16. A nibble q in the lowest bits of an FP16 0x6400 makes 1024 + q, one in the
// next four bits 1024 + 16 q, both exact. One mixed-precision fused multiply-add then gives (1024 + q) * scale -
// (1024 + zero) * scale, or (1024 + 16 q) * scale / 16 - (64 + zero) * scale. Every product is exact in FP32 (at most
// 19 significant bits), so the single rounding of the sum leaves the exact (q - zero) * scale, which is then rounded to
// BF16 as above: the weights are the same bits, for every scale from 2^-122 up (scale / 16 stays a normal number).
// Nibbles i and i + 4 of the word share one FP16 pair.
struct Fp16Terms {
    float scale;       // for a nibble in the lowest bits of an FP16
    float minus_bias;  // -(1024 + zero) * scale
    float scale16;     // scale / 16, for a nibble in the next four bits
    float minus_bias16;  // -(64 + zero) * scale
};

__device__ inline Fp16Terms fp16_terms(float scale, int zero) {
    return {scale, -static_cast<float>(1024 + zero) * scale, scale * 0.0625f, -static_cast<float>(64 + zero) * scale};
}

__device__ inline bf16x8 weight_word_fp16(uint32_t word, const Fp16Terms& terms) {
    const uint32_t high = word >> 8;
    const f16x2 pairs[4] = {
        __builtin_bit_cast(f16x2, (word & 0x000F000Fu) | 0x64006400u),  // nibbles 0 and 4
        __builtin_bit_cast(f16x2, (word & 0x00F000F0u) | 0x64006400u),  // nibbles 1 and 5, times 16
        __builtin_bit_cast(f16x2, (high & 0x000F000Fu) | 0x64006400u),  // nibbles 2 and 6
        __builtin_bit_cast(f16x2, (high & 0x00F000F0u) | 0x64006400u),  // nibbles 3 and 7, times 16
    };
    uint32_t r[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float scale = i % 2 ? terms.scale16 : terms.scale;
        const float minus_bias = i % 2 ? terms.minus_bias16 : terms.minus_bias;
        r[i] = rounded(__builtin_fmaf(static_cast<float>(pairs[i].x), scale, minus_bias));
        r[i + 4] = rounded(__builtin_fmaf(static_cast<float>(pairs[i].y), scale, minus_bias));
    }
    u32x4 weights;
    weights.x = __builtin_amdgcn_perm(r[1], r[0], 0x07060302u);
    weights.y = __builtin_amdgcn_perm(r[3], r[2], 0x07060302u);
    weights.z = __builtin_amdgcn_perm(r[5], r[4], 0x07060302u);
    weights.w = __builtin_amdgcn_perm(r[7], r[6], 0x07060302u);
    return __builtin_bit_cast(bf16x8, weights);
}

// Element i of a vector, read by name (see row_kernel).
__device__ inline uint32_t element(u32x4 v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// A workgroup barrier that orders shared memory only: global reads in flight stay in flight across it.
__device__ inline void shared_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// One row of x: a wave for each row of the weight. A lane takes one group of 32 columns per step: four packed words,
// one scale, one zero point and 32 values of x.
__global__ void __launch_bounds__(32 * kRowWaves)
    row_kernel(const uint32_t* __restrict__ packed, const uint16_t* __restrict__ scale,
               const uint32_t* __restrict__ zero, const uint32_t* __restrict__ x, uint16_t* __restrict__ y, int out,
               int in) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kRowWaves + (threadIdx.x >> 5);
    if (row >= out) {
        return;
    }
    const int groups = in / kGroup;
    const u32x4* words = reinterpret_cast<const u32x4*>(packed + static_cast<size_t>(row) * (in / 8));
    const u32x4* values = reinterpret_cast<const u32x4*>(x);
    const uint16_t* scales = scale + static_cast<size_t>(row) * groups;
    const uint32_t* zeros = zero ? zero + static_cast<size_t>(row >> 3) * groups : nullptr;
    const int shift = (row & 7) * 4;  // Where this row's zero point sits in a word.

    float sum = 0.0f;
    for (int group = lane; group < groups; group += 32) {
        // Each weight is read once, so it is read without being kept in the caches.
        const u32x4 packed_words = __builtin_nontemporal_load(words + group);
        const float scale_here = from_bits(static_cast<uint32_t>(scales[group]) << 16);
        // A symmetric layer stores its values offset by 8, which is a zero point of 8 in every row.
        const int zero_here = zeros ? static_cast<int>((zeros[group] >> shift) & 15u) : 8;
        // Vector elements are read by name: subscripting a vector held in an array gave wrong elements with this
        // compiler.
        const uint32_t word[4] = {packed_words.x, packed_words.y, packed_words.z, packed_words.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const u32x4 eight = values[group * 4 + i];  // Columns 8i to 8i + 7 of the group, as pairs.
            const uint32_t pairs[4] = {eight.x, eight.y, eight.z, eight.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bf16x2 weights = as_pair(weight_pair(word[i], j, zero_here, scale_here));
                sum = __builtin_amdgcn_fdot2_f32_bf16(weights, as_pair(pairs[j]), sum, false);
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 32);
    }
    if (lane == 0) {
        y[row] = static_cast<uint16_t>(rounded(sum) >> 16);
    }
}

// The dot product kernel on one row of x.
int one(const uint32_t* packed, const uint32_t* scale, const uint32_t* zero, const uint32_t* x, uint16_t* y, int out,
        int in) {
    hipLaunchKernelGGL(row_kernel, dim3((out + kRowWaves - 1) / kRowWaves), dim3(32 * kRowWaves), 0, 0, packed,
                       reinterpret_cast<const uint16_t*>(scale), zero, x, y, out, in);
    return hipGetLastError() == hipSuccess ? 0 : -3;
}

// The matrix instruction computes D += A x B on 16x16 matrices spread over the 32 lanes of a wave. Lane l and lane
// l + 16 hold row l of A, its first and last 8 inner positions; likewise column l of B; and column l of D, its first
// and last 8 rows. A is a weight tile and B the transposed tile of x, so a lane holds 8 columns of one weight row, 8
// columns of one row of x, and 8 outputs of that row of x.
//
// Within a run of 64 columns lanes 0-15 take the first 32 and lanes 16-31 the last 32, eight at a time over four
// products. A lane then reads one cache line of x per run, and the weights of one group of a row.
template <int T>
__global__ void tile_kernel(const uint32_t* __restrict__ packed, const uint32_t* __restrict__ scale,
                            const uint32_t* __restrict__ zero, const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
                            int out, int in, int rows) {
    // weights[(tile * kProducts + product) * 32 + lane]: the lanes of a wave read consecutive addresses. Addresses 64
    // bytes apart are served a few at a time.
    __shared__ bf16x8 weights[T * kProducts * 32];
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int wave = thread >> 5;
    const int half = lane >> 4;
    const int first = blockIdx.x * (T * kTile);
    const int row_of_x = wave * kTile + (lane & 15);
    // Lanes past the last row of x repeat it, and their sums are not written.
    const uint16_t* values = x + static_cast<size_t>(row_of_x < rows ? row_of_x : rows - 1) * in + half * (kRun / 2);

    floatx8 sum[T];
#pragma unroll
    for (int tile = 0; tile < T; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum[tile][i] = 0.0f;
        }
    }

    for (int start = 0; start < in; start += kUnit) {
        // A thread decodes one cache line of packed words: the four groups of 128 columns of one row.
        for (int task = thread; task < T * kTile * kLines; task += blockDim.x) {
            const int r = task % (T * kTile);
            const int line = task / (T * kTile);
            const int row = first + r;
            const uint32_t* words = packed + static_cast<size_t>(row) * (in / 8) + start / 8 + line * kLineWords;
            uint32_t w[kLineWords];
#pragma unroll
            for (int i = 0; i < kLineWords; ++i) {
                w[i] = words[i];
            }
            const uint32_t* scales =
                scale + static_cast<size_t>(row) * (in / kGroup / 2) + start / (kGroup * 2) + line * 2;
            const uint32_t s[2] = {scales[0], scales[1]};
            const uint32_t* zeros =
                zero ? zero + static_cast<size_t>(row >> 3) * (in / kGroup) + start / kGroup + line * 4 : nullptr;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float scale_here = from_bits(j % 2 ? s[j / 2] & 0xFFFF0000u : s[j / 2] << 16);
                // A symmetric layer stores its values offset by 8, which is a zero point of 8 in every row.
                const int zero_here = zeros ? static_cast<int>((zeros[j] >> ((row & 7) * 4)) & 15u) : 8;
                const int group = line * 4 + j;
                const int product = (r / kTile) * kProducts + (group >> 1) * 4;
                const int holder = (group & 1) * kTile + r % kTile;  // The lane that multiplies this group.
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    weights[(product + m) * 32 + holder] = weight_word(w[j * 4 + m], zero_here, scale_here);
                }
            }
        }
        __syncthreads();
        const auto multiply = [&](int run) {
            const bf16x8_unaligned* source = reinterpret_cast<const bf16x8_unaligned*>(values + start + run * kRun);
            bf16x8 b[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                b[m] = source[m];
            }
#pragma unroll
            for (int tile = 0; tile < T; ++tile) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const bf16x8 a = weights[(tile * kProducts + run * 4 + m) * 32 + lane];
                    sum[tile] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b[m], sum[tile]);
                }
            }
        };
        // Unrolled, the loop issues every load of the unit before the first product. With one weight tile, which is
        // the case of few waves, that hides the wait for memory. With two it runs slower than the loop.
        if (T == 1) {
#pragma unroll
            for (int run = 0; run < kUnit / kRun; ++run) {
                multiply(run);
            }
        } else {
#pragma nounroll
            for (int run = 0; run < kUnit / kRun; ++run) {
                multiply(run);
            }
        }
        // The next unit overwrites the weights only after every wave has used them.
        __syncthreads();
    }

    if (row_of_x < rows) {
#pragma unroll
        for (int tile = 0; tile < T; ++tile) {
            uint16_t* target = y + static_cast<size_t>(row_of_x) * out + first + tile * kTile + half * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                target[i] = static_cast<uint16_t>(rounded(sum[tile][i]) >> 16);
            }
        }
    }
}

// The matrix kernel: a wave for each tile of rows of x. A workgroup takes two weight tiles, or one.
int tiles(const uint32_t* packed, const uint32_t* scale, const uint32_t* zero, const uint16_t* x, uint16_t* y, int out,
          int in, int rows, bool two) {
    const int waves = (rows + kTile - 1) / kTile;
    if (two) {
        hipLaunchKernelGGL((tile_kernel<2>), dim3(out / (2 * kTile)), dim3(32 * waves), 0, 0, packed, scale, zero, x, y,
                           out, in, rows);
    } else {
        hipLaunchKernelGGL((tile_kernel<1>), dim3(out / kTile), dim3(32 * waves), 0, 0, packed, scale, zero, x, y, out,
                           in, rows);
    }
    return hipGetLastError() == hipSuccess ? 0 : -3;
}

// The matrix kernel reading ahead, for 48 to 128 rows of x: a workgroup takes two weight tiles. The tiles of x go in
// groups of MT, and KS waves share each group: wave w takes group w % groups and the runs
// [k * kMyRuns, (k + 1) * kMyRuns) of every unit, k = w / groups, kMyRuns = 8 / KS. Each weight fragment read from
// shared memory feeds MT products and each fragment of x two, so fewer bytes come into registers for each product, and
// the column split gives the workgroup more waves for the same shared memory. When the tiles of x do not fill the last
// group, its waves skip the empty tile. A workgroup has at least 128 threads, one for each 128-column line of its
// weight rows in a unit, or with HALF 256, one for each half line.
//
// Unit after unit, a wave
//   - reads the x of its runs of the unit into registers (with RING, the x of each run kSlots runs before its
//     products);
//   - waits for every wave to be done with the previous unit's weights in shared memory;
//   - decodes the packed words, scales and zero points of the unit, read during the previous unit, into shared memory;
//   - waits for the decode, then reads the next unit's packed words, scales and zero points (with RING, right after its
//     last read of the unit's x);
//   - multiplies.
// Reads complete in the order they were made, so x is read before the next unit's weights: the products wait for x and
// not for them. The sums of waves k = 1 .. KS - 1 of each group go through shared memory to wave k = 0, which adds them
// to its own in order of k.
template <int MT, int KS, bool RING, bool HALF, bool ROUND>
__global__ void __launch_bounds__(512)
    ahead_kernel(const uint32_t* __restrict__ packed, const uint32_t* __restrict__ scale,
                 const uint32_t* __restrict__ zero, const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int out,
                 int in, int rows) {
    constexpr int kMyRuns = kUnit / kRun / KS;            // Runs of a unit a wave takes.
    constexpr int kSlots = RING ? kMyRuns / 2 : kMyRuns;  // Runs of x a wave holds.
    constexpr int kTiles = 2;
    constexpr int kTaskGroups = HALF ? 2 : 4;  // Groups of a weight row a thread decodes in a unit.
    constexpr int kTasks = kTiles * kTile * kLines * 4 / kTaskGroups;
    static_assert(MT >= 2, "a group of one tile would hold no fewer in a last group");
    static_assert(!RING || kMyRuns >= 2, "a ring holds half of a wave's runs of a unit");
    __shared__ bf16x8 weights[kTiles * kProducts * 32];
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int wave = thread >> 5;
    const int half = lane >> 4;
    const int groups = ((rows + kTile - 1) / kTile + MT - 1) / MT;
    const int part = wave / groups;
    const int group_of_x = wave % groups;
    const int first = blockIdx.x * (kTiles * kTile);
    const int units = in / kUnit;
    // With ROUND each workgroup starts at a unit of its own and goes round the row (see `ahead_launch`).
    const int start = ROUND ? static_cast<int>(blockIdx.x % units) : 0;
    const auto unit_at = [&](int i) __attribute__((always_inline)) {
        if constexpr (ROUND) {
            return i + start < units ? i + start : i + start - units;
        } else {
            return i;
        }
    };
    // Lanes past the last row of x repeat it, and their sums are not written.
    const bf16x8* values[MT];
#pragma unroll
    for (int p = 0; p < MT; ++p) {
        const int row_of_x = (group_of_x * MT + p) * kTile + (lane & 15);
        const size_t row_start = static_cast<size_t>(row_of_x < rows ? row_of_x : rows - 1) * in;
        values[p] = reinterpret_cast<const bf16x8*>(x + row_start + half * (kRun / 2) + part * kMyRuns * kRun);
    }

    // The decode task of this thread: a cache line of packed words, four groups of one weight row, or with HALF half of
    // one, two groups.
    const bool decodes = thread < kTasks;
    const int r = thread % (kTiles * kTile);
    const int line = thread / (kTiles * kTile) / (HALF ? 2 : 1);
    const int sub = HALF ? thread / (kTiles * kTile) % 2 : 0;  // With HALF, which half of the line.
    const int row = first + r;
    const uint32_t* words_of_row = packed + static_cast<size_t>(row) * (in / 8) + line * kLineWords + sub * 8;
    const uint32_t* scales_of_row = scale + static_cast<size_t>(row) * (in / kGroup / 2) + line * 2 + sub;
    const uint32_t* zeros_of_row = zero ? zero + static_cast<size_t>(row >> 3) * (in / kGroup) + line * 4 + sub * 2
                                        : nullptr;
    const int shift = (row & 7) * 4;  // Where this row's zero point sits in a word.
    u32x4 words[4];
    u32x2 scales;  // With HALF, .x holds the scales of the two groups.
    u32x4 zeros;   // With HALF, .x and .y hold the zero points' words of the two groups.
    const auto read_weights = [&](int unit) __attribute__((always_inline)) {
        if (decodes) {
            const u32x4* source = reinterpret_cast<const u32x4*>(words_of_row + unit * (kUnit / 8));
#pragma unroll
            for (int j = 0; j < kTaskGroups; ++j) {
                words[j] = source[j];
            }
            if constexpr (HALF) {
                scales.x = scales_of_row[unit * (kUnit / kGroup / 2)];
                const uint32_t* zeros_here = zeros_of_row + unit * (kUnit / kGroup);
                const u32x2 pair = zeros_of_row ? *reinterpret_cast<const u32x2*>(zeros_here) : u32x2{0, 0};
                zeros.x = pair.x;
                zeros.y = pair.y;
            } else {
                scales = *reinterpret_cast<const u32x2*>(scales_of_row + unit * (kUnit / kGroup / 2));
                zeros = zeros_of_row ? *reinterpret_cast<const u32x4*>(zeros_of_row + unit * (kUnit / kGroup))
                                     : u32x4{0, 0, 0, 0};
            }
        }
    };
    const auto decode = [&]() __attribute__((always_inline)) {
        if (decodes) {
#pragma unroll
            for (int j = 0; j < kTaskGroups; ++j) {
                const uint32_t pair = j < 2 ? scales.x : scales.y;
                const float scale_here = from_bits(j % 2 ? pair & 0xFFFF0000u : pair << 16);
                // A symmetric layer stores its values offset by 8, which is a zero point of 8 in every row.
                const int zero_here = zeros_of_row ? static_cast<int>((element(zeros, j) >> shift) & 15u) : 8;
                const Fp16Terms terms = fp16_terms(scale_here, zero_here);
                const int group = line * 4 + sub * 2 + j;
                const int product = (r / kTile) * kProducts + (group >> 1) * 4;
                const int holder = (group & 1) * kTile + r % kTile;  // The lane that multiplies this group.
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    weights[(product + m) * 32 + holder] = weight_word_fp16(element(words[j], m), terms);
                }
            }
        }
    };
    // The x of run `run` of this wave's runs of `unit`, for the first `held` tiles of its group, into a slot of
    // registers. The unit is the same for every lane and the run a constant, so the address is the lane's row plus a
    // uniform offset.
    bf16x8 b[MT][kSlots][4];
    const auto read_x = [&](int unit, int run, int slot, auto held) __attribute__((always_inline)) {
#pragma unroll
        for (int p = 0; p < decltype(held)::value; ++p) {
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                b[p][slot][m] = values[p][(unit * kUnit + run * kRun) / 8 + m];
            }
        }
    };

    floatx8 sum[kTiles][MT];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int p = 0; p < MT; ++p) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sum[tile][p][i] = 0.0f;
            }
        }
    }
    read_weights(start);
    // The units, for a wave that holds `held` tiles of x: all MT of its group, or one fewer in a last group the tiles
    // do not fill. The wave chooses once, before the loop: a choice inside it makes the compiler schedule every wave's
    // loop worse.
    const auto all_units = [&](auto held) __attribute__((always_inline)) {
        if constexpr (RING) {
#pragma unroll
            for (int slot = 0; slot < kSlots; ++slot) {
                read_x(start, slot, slot, held);
            }
        }
        for (int i = 0; i < units; ++i) {
            const int unit = unit_at(i);
            const int next = i + 1 < units ? unit_at(i + 1) : unit;
            if constexpr (!RING) {
                // The x of the wave's runs of the unit, all at once.
#pragma unroll
                for (int run = 0; run < kMyRuns; ++run) {
                    read_x(unit, run, run, held);
                }
            }
            if (i > 0) {
                shared_barrier();
            }
            decode();
            shared_barrier();
            if constexpr (!RING) {
                read_weights(next);
            }
#pragma unroll
            for (int run = 0; run < kMyRuns; ++run) {
                const int slot = run % kSlots;
                const auto product = [&](int tile, int m) __attribute__((always_inline)) {
                    const bf16x8 a = weights[(tile * kProducts + (part * kMyRuns + run) * 4 + m) * 32 + lane];
#pragma unroll
                    for (int p = 0; p < decltype(held)::value; ++p) {
                        sum[tile][p] =
                            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b[p][slot][m], sum[tile][p]);
                    }
                };
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int tile = 0; tile < kTiles; ++tile) {
                        product(tile, m);
                    }
                }
                if constexpr (RING) {
                    // The run kSlots ahead, in this unit or the next (the last unit reads its own runs again).
                    if (run + kSlots < kMyRuns) {
                        read_x(unit, run + kSlots, slot, held);
                    } else {
                        read_x(next, run + kSlots - kMyRuns, slot, held);
                    }
                    if (run == kMyRuns - kSlots - 1) {
                        read_weights(next);
                    }
                }
            }
        }
    };
    if ((group_of_x * MT + MT - 1) * kTile < rows) {
        all_units(std::integral_constant<int, MT>{});
    } else {
        all_units(std::integral_constant<int, MT - 1>{});
    }

    // The weights are no longer needed: their shared memory holds the other waves' sums, one weight tile at a time. For
    // every form `ahead` picks, the (KS - 1) * groups * MT fragments of 1 KB fit in its 32 KB.
    floatx8* partial = reinterpret_cast<floatx8*>(weights);
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
        shared_barrier();
        if (part > 0) {
#pragma unroll
            for (int p = 0; p < MT; ++p) {
                partial[(((part - 1) * groups + group_of_x) * MT + p) * 32 + lane] = sum[tile][p];
            }
        }
        shared_barrier();
        if (part == 0) {
#pragma unroll
            for (int k = 1; k < KS; ++k) {
#pragma unroll
                for (int p = 0; p < MT; ++p) {
                    sum[tile][p] += partial[(((k - 1) * groups + group_of_x) * MT + p) * 32 + lane];
                }
            }
        }
    }
    if (part > 0) {
        return;
    }
#pragma unroll
    for (int p = 0; p < MT; ++p) {
        const int row_of_x = (group_of_x * MT + p) * kTile + (lane & 15);
        if (row_of_x < rows) {
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                uint16_t* target = y + static_cast<size_t>(row_of_x) * out + first + tile * kTile + half * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    target[i] = static_cast<uint16_t>(rounded(sum[tile][p][i]) >> 16);
                }
            }
        }
    }
}

// When a row of packed words is a multiple of 4 KB (in a multiple of 8192), the workgroups that read the same unit at
// once read addresses on the same memory channels, so each of them starts at a unit of its own. At a multiple of 1 KB
// this pays only for the forms without a ring of x: with a ring, the workgroups that read the same x at once share it
// in the caches, which is worth more.
template <int MT, int KS, bool RING, bool HALF = false>
int ahead_launch(const uint32_t* packed, const uint32_t* scale, const uint32_t* zero, const uint16_t* x, uint16_t* y,
                 int out, int in, int rows) {
    const int groups = ((rows + kTile - 1) / kTile + MT - 1) / MT;
    const dim3 grid(out / (2 * kTile));
    const dim3 threads(32 * KS * groups);
    if (in % 8192 == 0 || (!RING && in % 2048 == 0)) {
        hipLaunchKernelGGL((ahead_kernel<MT, KS, RING, HALF, true>), grid, threads, 0, 0, packed, scale, zero, x, y,
                           out, in, rows);
    } else {
        hipLaunchKernelGGL((ahead_kernel<MT, KS, RING, HALF, false>), grid, threads, 0, 0, packed, scale, zero, x, y,
                           out, in, rows);
    }
    return hipGetLastError() == hipSuccess ? 0 : -3;
}

// The matrix kernel reading ahead, in the form measured the fastest for each number of tiles of x: for three tiles one
// group of three with eight waves; from four, groups of two with four waves to each and half lines to decode, from
// five with x in a ring.
int ahead(const uint32_t* packed, const uint32_t* scale, const uint32_t* zero, const uint16_t* x, uint16_t* y, int out,
          int in, int rows) {
    const int tiles_of_x = (rows + kTile - 1) / kTile;
    if (tiles_of_x <= 3) {
        return ahead_launch<3, 8, false>(packed, scale, zero, x, y, out, in, rows);
    }
    if (tiles_of_x == 4) {
        return ahead_launch<2, 4, false, true>(packed, scale, zero, x, y, out, in, rows);
    }
    return ahead_launch<2, 4, true, true>(packed, scale, zero, x, y, out, in, rows);
}

int refused(rdna4_w4a16::dev::Reason reason) { return -static_cast<int>(reason); }

__global__ void nothing_kernel() {}

}  // namespace

void rdna4_w4a16::dev::launch_nothing() { hipLaunchKernelGGL(nothing_kernel, dim3(1), dim3(1), 0, 0); }

extern "C" {

// The path a product of `rows` rows takes through a layer of `out` by `in` weights in groups of `group` columns: a
// `rdna4_w4a16::dev::Path` above zero, or minus the `rdna4_w4a16::dev::Reason` why the checkpoint's own layer runs it. `dtype`
// is the input's `rdna4_w4a16::dev::Dtype`, and `bias` is whether the layer adds one.
RDNA4_W4A16_API int w4a16_route(int out, int in, int group, int rows, int dtype, int bias) {
    using namespace rdna4_w4a16::dev;
    if (dtype < RDNA4_W4A16_DTYPE_OTHER || dtype > RDNA4_W4A16_DTYPE_F32 || (bias != 0 && bias != 1)) {
        return RDNA4_W4A16_STATUS_ARGUMENT;
    }
    if (dtype != static_cast<int>(Dtype::BF16)) {
        return refused(Reason::Dtype);
    }
    if (bias != 0) {
        return refused(Reason::Bias);
    }
    if (group != kGroup || in < kUnit || in % kUnit != 0 || out < kTile || out % kTile != 0) {
        return refused(Reason::Shape);
    }
    if (rows < 1 || rows > kRows) {
        return refused(Reason::Rows);
    }
    if (rows == 1) {
        return static_cast<int>(Path::Dot);
    }
    return static_cast<int>(rows >= kTwoTiles && out % (2 * kTile) == 0 ? Path::Tile2 : Path::Tile1);
}

// y [rows, out] = x [rows, in] times the transposed weight, both BF16 and without a bias, on the current device and
// its default stream. Returns 0, or minus the `rdna4_w4a16::dev::Reason` why y holds nothing to use: the layer or the
// number of rows is not one a kernel takes, or a launch failed.
RDNA4_W4A16_API int w4a16_linear(const uint32_t* packed, const uint16_t* scale, const uint32_t* zero,
                                       const uint16_t* x, uint16_t* y, int out, int in, int group, int rows) {
    using namespace rdna4_w4a16::dev;
    const int route = w4a16_route(out, in, group, rows, static_cast<int>(Dtype::BF16), 0);
    if (route < 0) {
        return route;
    }
    const auto aligned = [](const void* pointer, uintptr_t alignment) {
        return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
    };
    if (!aligned(packed, 16) || !aligned(scale, 4) || (zero && !aligned(zero, 4)) ||
        !aligned(x, 16) || !aligned(y, 2)) {
        return RDNA4_W4A16_STATUS_ARGUMENT;
    }
    const Path path = static_cast<Path>(route);
    const Timed timed({Op::Int4, path, Reason::None, rows, in, out, Dtype::BF16});
    const uint32_t* scale_words = reinterpret_cast<const uint32_t*>(scale);
    int code = 0;
    if (path == Path::Dot) {
        code = one(packed, scale_words, zero, reinterpret_cast<const uint32_t*>(x), y, out, in);
    } else if (path == Path::Tile2 && rows <= kAheadLast) {
        code = ahead(packed, scale_words, zero, x, y, out, in, rows);
    } else {
        code = tiles(packed, scale_words, zero, x, y, out, in, rows, path == Path::Tile2);
    }
    return code == 0 ? 0 : refused(Reason::Launch);
}

}  // extern "C"
