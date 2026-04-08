// nms_kernel.cu – CUDA-accelerated Non-Maximum Suppression
// Enabled only when building with CUDA support (CUDA_FOUND in CMake).
#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace adas::cuda {

// ─── Helper: IoU between two boxes stored as (x1,y1,x2,y2) ─────────────────
__device__ inline float boxIou(
    float ax1, float ay1, float ax2, float ay2,
    float bx1, float by1, float bx2, float by2)
{
    float ix1 = fmaxf(ax1, bx1), iy1 = fmaxf(ay1, by1);
    float ix2 = fminf(ax2, bx2), iy2 = fminf(ay2, by2);
    float inter = fmaxf(0.f, ix2 - ix1) * fmaxf(0.f, iy2 - iy1);
    float aArea = (ax2-ax1)*(ay2-ay1);
    float bArea = (bx2-bx1)*(by2-by1);
    float uni   = aArea + bArea - inter;
    return (uni > 0.f) ? inter / uni : 0.f;
}

/// Each thread processes one (reference_box, candidate_box) pair and sets
/// suppressMask[i] = 1 if candidate i is suppressed by the reference box.
///
/// @param boxes          flat array [N, 4]: (x1,y1,x2,y2)
/// @param scores         score per box     [N]
/// @param suppressMask   output mask       [N]
/// @param refIdx         index of the reference (highest-score) box
/// @param nBoxes         total number of boxes
/// @param iouThreshold   IoU threshold above which a box is suppressed
__global__ void nmsKernel(
    const float* __restrict__ boxes,
    const float* __restrict__ /*scores*/,
    uint8_t*                  suppressMask,
    int                       refIdx,
    int                       nBoxes,
    float                     iouThreshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBoxes || i == refIdx || suppressMask[i]) return;

    float ax1 = boxes[refIdx*4+0], ay1 = boxes[refIdx*4+1];
    float ax2 = boxes[refIdx*4+2], ay2 = boxes[refIdx*4+3];
    float bx1 = boxes[i*4+0],      by1 = boxes[i*4+1];
    float bx2 = boxes[i*4+2],      by2 = boxes[i*4+3];

    if (boxIou(ax1,ay1,ax2,ay2, bx1,by1,bx2,by2) > iouThreshold)
        suppressMask[i] = 1;
}

/// Host-side launcher for GPU NMS.
void launchNMS(
    const float* d_boxes, const float* d_scores,
    uint8_t* d_mask, int nBoxes, float iouThreshold,
    const int* sortedIndices, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    for (int i = 0; i < nBoxes; ++i) {
        int refIdx = sortedIndices[i];
        // Check if ref is already suppressed (requires host-side mask read –
        // in production use a prefix-scan; simplified here for clarity)
        nmsKernel<<<(nBoxes + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
            d_boxes, d_scores, d_mask, refIdx, nBoxes, iouThreshold);
    }
}

} // namespace adas::cuda
#endif // __CUDACC__
