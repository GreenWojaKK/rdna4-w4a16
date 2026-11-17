// The developer mode of the device kernels. It records what each linear product was, M rows of input by K inputs by N
// outputs and its dtype, which path it took, and how long the device spent on it, grouped by spans the caller opens
// and names. Spans of one name form a phase.
//
// The numbers below are the record format's. `record/src/spec.rs` holds the same names and is what readers use.
#pragma once

#include "api.h"

namespace rdna4_w4a16::dev {

enum class Op : int { Int4 = RDNA4_W4A16_OP_INT4, Head = RDNA4_W4A16_OP_HEAD };

// Unpack is the checkpoint's own path: its kernel unpacks the whole weight to BF16 and the product follows.
enum class Path : int { Unpack = RDNA4_W4A16_PATH_UNPACK, Dot = RDNA4_W4A16_PATH_DOT, Tile1 = RDNA4_W4A16_PATH_TILE1, Tile2 = RDNA4_W4A16_PATH_TILE2, Dense = RDNA4_W4A16_PATH_DENSE };

// Why an int4 product took the checkpoint's path.
enum class Reason : int { None = RDNA4_W4A16_REASON_NONE, Rows = RDNA4_W4A16_REASON_ROWS, Dtype = RDNA4_W4A16_REASON_DTYPE, Bias = RDNA4_W4A16_REASON_BIAS, Shape = RDNA4_W4A16_REASON_SHAPE, Launch = RDNA4_W4A16_REASON_LAUNCH };
inline constexpr int kReasons = 6;

enum class Dtype : int { Other = RDNA4_W4A16_DTYPE_OTHER, BF16 = RDNA4_W4A16_DTYPE_BF16, F16 = RDNA4_W4A16_DTYPE_F16, F32 = RDNA4_W4A16_DTYPE_F32 };

struct Product {
    Op op;
    Path path;
    Reason reason;
    int m;  // Rows of input.
    int k;  // Inputs of a row.
    int n;  // Outputs of a row.
    Dtype dtype;
};

// Whether the developer mode records.
bool on();

// The start and the end of one product on the device's default stream. Products do not nest: one that starts inside
// another is part of it.
void begin(const Product& product);
void end();

// A launch on the default stream that computes nothing. The profiler measures with it what events cost.
void launch_nothing();

// Times a product for as long as it is in scope.
class Timed {
  public:
    explicit Timed(const Product& product) : active_(on()) {
        if (active_) {
            begin(product);
        }
    }
    ~Timed() {
        if (active_) {
            end();
        }
    }
    Timed(const Timed&) = delete;
    Timed& operator=(const Timed&) = delete;

  private:
    bool active_;
};

}  // namespace rdna4_w4a16::dev
