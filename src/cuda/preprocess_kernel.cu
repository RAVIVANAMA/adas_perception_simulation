// preprocess_kernel.cu – GPU-accelerated image preprocessing for inference.
// Performs letterbox resize + BGR→RGB + [0,1] normalisation in one kernel pass.
#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cstdint>

namespace adas::cuda {

/// Letterbox + normalise kernel.
/// Input:  BGR uint8 image  [srcH, srcW, 3]
/// Output: float CHW tensor [3, dstH, dstW]  (values in [0, 1])
///
/// @param src        device pointer to BGRU8 source image
/// @param dst        device pointer to float CHW output tensor
/// @param srcW,srcH  source dimensions
/// @param dstW,dstH  target (model input) dimensions
/// @param scaleInv   1 / letterbox_scale so we can map dst → src
/// @param padX,padY  letterbox padding (pixels in dst space)
__global__ void preprocessKernel(
    const uint8_t* __restrict__ src,
    float*                      dst,
    int srcW, int srcH,
    int dstW, int dstH,
    float scaleInv,
    int padX, int padY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    // Map destination pixel → source pixel
    int srcX = __float2int_rd((x - padX) * scaleInv);
    int srcY = __float2int_rd((y - padY) * scaleInv);

    // Fill letterbox padding with grey (114/255)
    if (srcX < 0 || srcX >= srcW || srcY < 0 || srcY >= srcH) {
        int base = y * dstW + x;
        dst[0 * dstH * dstW + base] = 114.f / 255.f;
        dst[1 * dstH * dstW + base] = 114.f / 255.f;
        dst[2 * dstH * dstW + base] = 114.f / 255.f;
        return;
    }

    int si   = (srcY * srcW + srcX) * 3;
    int base = y * dstW + x;

    dst[0 * dstH * dstW + base] = src[si + 2] / 255.f;  // R
    dst[1 * dstH * dstW + base] = src[si + 1] / 255.f;  // G
    dst[2 * dstH * dstW + base] = src[si + 0] / 255.f;  // B
}

/// Host-side launcher.
void launchPreprocess(
    const uint8_t* d_src, float* d_dst,
    int srcW, int srcH,
    int dstW, int dstH,
    float scale,          // letterbox scale (min of W-ratio, H-ratio)
    int padX, int padY,
    cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x,
              (dstH + block.y - 1) / block.y);
    preprocessKernel<<<grid, block, 0, stream>>>(
        d_src, d_dst,
        srcW, srcH, dstW, dstH,
        1.f / scale, padX, padY);
}

} // namespace adas::cuda
#endif // __CUDACC__
