#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_half, 4, 256, true, false, QKVLayout::kNHD, PosEncodingMode::kALiBi)
