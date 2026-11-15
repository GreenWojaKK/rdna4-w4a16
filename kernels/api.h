// Marks the functions of the library's C ABI. A Windows DLL exports only the functions marked for export; a Linux
// shared library exports every function it defines. The mark is for the host code; device code has no exports.
#pragma once

#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)
#define RDNA4_W4A16_API __declspec(dllexport)
#else
#define RDNA4_W4A16_API
#endif
