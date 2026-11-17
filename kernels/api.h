// Public C ABI. Requires only the C standard library, not the HIP SDK.
#ifndef RDNA4_W4A16_API_H
#define RDNA4_W4A16_API_H
#include <stdint.h>

#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)
#if defined(RDNA4_W4A16_BUILD)
#define RDNA4_W4A16_API __declspec(dllexport)
#else
#define RDNA4_W4A16_API __declspec(dllimport)
#endif
#else
#define RDNA4_W4A16_API __attribute__((visibility("default")))
#endif

#define RDNA4_W4A16_ABI_VERSION 2

// Parameters use fixed-width integers; enums supply constants, not enum-sized parameters.
enum rdna4_w4a16_status {
    RDNA4_W4A16_STATUS_OK = 0,
    RDNA4_W4A16_STATUS_ROWS = -1,
    RDNA4_W4A16_STATUS_DTYPE = -2,
    RDNA4_W4A16_STATUS_BIAS = -3,
    RDNA4_W4A16_STATUS_SHAPE = -4,
    RDNA4_W4A16_STATUS_LAUNCH = -5,
    RDNA4_W4A16_STATUS_ARGUMENT = -6,
    RDNA4_W4A16_STATUS_STATE = -7,
    RDNA4_W4A16_STATUS_IO = -8,
    RDNA4_W4A16_STATUS_DEVICE = -9
};
enum rdna4_w4a16_path {
    RDNA4_W4A16_PATH_UNPACK = 0,
    RDNA4_W4A16_PATH_DOT = 1,
    RDNA4_W4A16_PATH_TILE1 = 2,
    RDNA4_W4A16_PATH_TILE2 = 3,
    RDNA4_W4A16_PATH_DENSE = 4
};
enum rdna4_w4a16_reason {
    RDNA4_W4A16_REASON_NONE = 0,
    RDNA4_W4A16_REASON_ROWS = 1,
    RDNA4_W4A16_REASON_DTYPE = 2,
    RDNA4_W4A16_REASON_BIAS = 3,
    RDNA4_W4A16_REASON_SHAPE = 4,
    RDNA4_W4A16_REASON_LAUNCH = 5
};
enum rdna4_w4a16_dtype {
    RDNA4_W4A16_DTYPE_OTHER = 0,
    RDNA4_W4A16_DTYPE_BF16 = 1,
    RDNA4_W4A16_DTYPE_F16 = 2,
    RDNA4_W4A16_DTYPE_F32 = 3
};
enum rdna4_w4a16_op {
    RDNA4_W4A16_OP_INT4 = 0,
    RDNA4_W4A16_OP_HEAD = 1
};

#ifdef __cplusplus
extern "C" {
#endif

RDNA4_W4A16_API uint32_t w4a16_abi_version(void);
// Static English text, valid for the lifetime of the library.
RDNA4_W4A16_API const char* w4a16_status_string(int32_t status);
// Pure host query: positive PATH_DOT/TILE1/TILE2, or a negative status. Bias is 0 or 1.
// ROWS/DTYPE/BIAS/SHAPE tell the caller to use its reference path.
RDNA4_W4A16_API int32_t w4a16_route(int32_t out, int32_t in, int32_t group, int32_t rows, int32_t dtype, int32_t bias);

// y = x W^T. Each weight is BF16((nibble - zero_point) * scale), accumulated in FP32.
// Contiguous device buffers on the current HIP device, owned by the caller:
// packed uint32[out,in/8] (16-byte aligned); scale BF16[out,in/32] (4-byte aligned);
// zero uint32[out/8,in/32] (4-byte aligned), or NULL for a zero point of 8;
// x BF16[rows,in] (16-byte aligned); y BF16[rows,out] (2-byte aligned).
// uint16_t holds BF16 bits. y must not overlap inputs. The caller supplies enough storage.
// All launches use the current device's DEFAULT stream. Keep buffers alive until it completes;
// order other-stream producers/consumers explicitly. Success means enqueued, not complete.
// Returns STATUS_OK or a negative status; on failure y is unusable. LAUNCH reports enqueue
// failure; the caller checks asynchronous errors when synchronizing the stream.
RDNA4_W4A16_API int32_t w4a16_linear(const uint32_t* packed, const uint16_t* scale, const uint32_t* zero, const uint16_t* x, uint16_t* y, int32_t out, int32_t in, int32_t group, int32_t rows);

// One process-global recorder, driven by one host thread on one unchanged current device.
// Serialize all library calls while recording. Spans do not nest. Products may nest; only the
// outer product is counted. Balance each successful begin with end before closing a span.
// All recorder functions return STATUS_OK or a negative status from the same status enum.
// start opens/truncates a UTF-8 path; every >= 0 samples INT4 timing, zero disables sampling.
RDNA4_W4A16_API int32_t w4a16_record_begin(const char* path, int32_t every);
// name is nonempty UTF-8; rows and tokens are nonnegative. Strings are consumed during the call.
RDNA4_W4A16_API int32_t w4a16_span_begin(const char* name, int64_t rows, int64_t tokens);
// Waits for the default stream to reach the span's end and writes its record.
RDNA4_W4A16_API int32_t w4a16_span_end(void);
// Caller-run operations: HEAD/DENSE/NONE or INT4/UNPACK with a nonzero fallback reason.
// Dimensions are positive. With recording off, valid op_begin/op_end calls are no-ops.
RDNA4_W4A16_API int32_t w4a16_op_begin(int32_t op, int32_t path, int32_t reason, int32_t m, int32_t k, int32_t n, int32_t dtype);
RDNA4_W4A16_API int32_t w4a16_op_end(void);
// Closes an open span and the file. *summary is library-owned UTF-8 JSON until the next
// start; copy it before then. Off: STATUS_STATE and *summary = NULL. Output is required.
RDNA4_W4A16_API int32_t w4a16_record_end(const char** summary);

#ifdef __cplusplus
}
#endif
#endif
