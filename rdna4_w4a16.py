"""Checked Python binding for the C ABI. Buffer arguments are device addresses.

Use the current HIP device's default stream and keep buffers alive until it completes.
The recorder is process-global and all calls must be serialized while it is active.
"""
import ctypes
import json
import os

from _rdna4_w4a16_abi import ABI_VERSION, SIGNATURES, Dtype, Op, Path, Reason, Status


class Library:
    """Loads and checks the ABI before binding functions; does not import PyTorch."""

    def __init__(self, path):
        self.raw = ctypes.CDLL(os.fspath(path))
        try:
            version = self.raw.w4a16_abi_version
        except AttributeError as error:
            raise RuntimeError("kernel library has no ABI version; rebuild it with its Python binding") from error
        version.argtypes, version.restype = [], ctypes.c_uint32
        found = version()
        if found != ABI_VERSION:
            raise RuntimeError(f"kernel ABI {found}, binding ABI {ABI_VERSION}; rebuild the library and binding together")
        for name, (result, args) in SIGNATURES.items():
            function = getattr(self.raw, name)
            function.argtypes, function.restype = args, result

    def check(self, status, operation):
        if status != Status.OK:
            message = self.raw.w4a16_status_string(status).decode("utf-8")
            raise RuntimeError(f"{operation}: {message} ({status})")
        return status

    def route(self, out, inputs, group, rows, dtype, bias):
        result = self.raw.w4a16_route(out, inputs, group, rows, dtype, int(bias))
        if result == Status.ARGUMENT:
            self.check(result, "route")
        return result

    def linear(self, packed, scale, zero, x, y, out, inputs, group, rows):
        return self.check(self.raw.w4a16_linear(
            packed, scale, zero, x, y, out, inputs, group, rows), "linear")

    def record_begin(self, path, every):
        self.check(self.raw.w4a16_record_begin(os.fspath(path).encode("utf-8"), every), "start recorder")

    def span_begin(self, name, rows, tokens):
        self.check(self.raw.w4a16_span_begin(name.encode("utf-8"), rows, tokens), "begin span")

    def span_end(self):
        self.check(self.raw.w4a16_span_end(), "end span")

    def op_begin(self, op, path, reason, m, k, n, dtype):
        self.check(self.raw.w4a16_op_begin(op, path, reason, m, k, n, dtype), "begin product")

    def op_end(self):
        self.check(self.raw.w4a16_op_end(), "end product")

    def record_end(self):
        summary = ctypes.c_char_p()
        self.check(self.raw.w4a16_record_end(ctypes.byref(summary)), "finish recorder")
        return json.loads(summary.value.decode("utf-8"))
