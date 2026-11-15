// The developer mode of the device kernels. It records what each linear product was, M rows of input by K inputs by N
// outputs and its dtype, which path it took, and how long the device spent on it, grouped by spans the caller opens
// and names. Spans of one name form a phase.
//
// The numbers below are the record format's. `record/src/spec.rs` holds the same names and is what readers use.
#pragma once

namespace rdna4_w4a16::dev {

enum class Op : int { Int4 = 0, Head = 1 };

// Unpack is the checkpoint's own path: its kernel unpacks the whole weight to BF16 and the product follows.
enum class Path : int { Unpack = 0, Dot = 1, Tile1 = 2, Tile2 = 3, Dense = 4 };

// Why an int4 product took the checkpoint's path.
enum class Reason : int { None = 0, Rows = 1, Dtype = 2, Bias = 3, Shape = 4, Launch = 5 };
inline constexpr int kReasons = 6;

enum class Dtype : int { Other = 0, BF16 = 1, F16 = 2, F32 = 3 };

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
