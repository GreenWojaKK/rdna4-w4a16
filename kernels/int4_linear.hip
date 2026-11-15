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
// Two kernels share the work by the number of rows of x.
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
// `rdna4_w4a16_route` chooses among them, or says why a product is the checkpoint's own layer's to run. In the
// developer mode every product is recorded with the path it took (see instrument.h).

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

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
constexpr int kSharedRows = 48;   // Up to here a workgroup takes one weight tile, beyond it two.

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
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

int refused(rdna4_w4a16::dev::Reason reason) { return -static_cast<int>(reason); }

__global__ void nothing_kernel() {}

}  // namespace

void rdna4_w4a16::dev::launch_nothing() { hipLaunchKernelGGL(nothing_kernel, dim3(1), dim3(1), 0, 0); }

extern "C" {

// The path a product of `rows` rows takes through a layer of `out` by `in` weights in groups of `group` columns: a
// `rdna4_w4a16::dev::Path` above zero, or minus the `rdna4_w4a16::dev::Reason` why the checkpoint's own layer runs it. `dtype`
// is the input's `rdna4_w4a16::dev::Dtype`, and `bias` is whether the layer adds one.
RDNA4_W4A16_API int rdna4_w4a16_route(int out, int in, int group, int rows, int dtype, int bias) {
    using namespace rdna4_w4a16::dev;
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
    return static_cast<int>(rows > kSharedRows && out % (2 * kTile) == 0 ? Path::Tile2 : Path::Tile1);
}

// y [rows, out] = x [rows, in] times the transposed weight, both BF16 and without a bias, on the current device and
// its default stream. Returns 0, or minus the `rdna4_w4a16::dev::Reason` why y holds nothing to use: the layer or the
// number of rows is not one a kernel takes, or a launch failed.
RDNA4_W4A16_API int rdna4_w4a16_linear(const uint32_t* packed, const uint32_t* scale, const uint32_t* zero,
                                       const uint32_t* x, uint16_t* y, int out, int in, int group, int rows) {
    using namespace rdna4_w4a16::dev;
    const int route = rdna4_w4a16_route(out, in, group, rows, static_cast<int>(Dtype::BF16), 0);
    if (route < 0) {
        return route;
    }
    const Path path = static_cast<Path>(route);
    const Timed timed({Op::Int4, path, Reason::None, rows, in, out, Dtype::BF16});
    const int code = path == Path::Dot ? one(packed, scale, zero, x, y, out, in)
                                       : tiles(packed, scale, zero, reinterpret_cast<const uint16_t*>(x), y, out, in, rows,
                                               path == Path::Tile2);
    return code == 0 ? 0 : refused(Reason::Launch);
}

}  // extern "C"
