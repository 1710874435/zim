#include "JSLAB_Config.h"
#ifdef JSLAB_USE_CUDA

#include "JSLAB_PhaseShiftDecoder_V2_GPU.h"
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// 3x3 pre-blur kernel weights (outer product of getGaussianKernel(3, sigma)).
__constant__ double c_blur[9];

// ---------------------------------------------------------------------------
// Device helper: 3x3 Gaussian blur of a single frame sample (BORDER_REPLICATE),
// computed on the fly so we never have to store the blurred frame stack.
// Inputs are raw pixel values (exact as float for 8/16-bit images); the blur
// accumulates in double to match cv::GaussianBlur run in double on the CPU.
// ---------------------------------------------------------------------------
__device__ __forceinline__ double blurredSample(
    const unsigned char* __restrict__ raw_frame, int irow, int jcol, int rows, int cols)
{
    double acc = 0.0;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = min(max(irow + dy, 0), rows - 1);
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = min(max(jcol + dx, 0), cols - 1);
            acc += c_blur[(dy + 1) * 3 + (dx + 1)] * (double)raw_frame[yy * cols + xx];
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Per-pixel multi-frequency phase-shift decode (3 frequencies, heterodyne
// unwrapping).  Mirrors JSLAB_MultiFreqPhaseShiftDecodeBodyV2::operator().
// The 3x3 pre-blur is fused in (blurredSample) so only the raw frame stack
// lives in device memory.  All phase math is double precision to keep the
// unwrap order (round) decisions identical to the CPU path.
// ---------------------------------------------------------------------------
__global__ void decodeKernel(
    const unsigned char* __restrict__ d_raw, int rows, int cols,
    const double* __restrict__ d_sin, const double* __restrict__ d_cos,
    int sn0, int sn1, int sn2,
    int base0, int base1, int base2,
    int so0, int so1, int so2,
    double f0, double f1, double f2,
    double mod_thresh, int avg_three,
    double* __restrict__ d_phase, int* __restrict__ d_k, double* __restrict__ d_mod)
{
    int jcol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || jcol >= cols)
        return;

    const int pix = irow * cols + jcol;
    const int frame_stride = rows * cols;

    const int    sn[3]   = { sn0, sn1, sn2 };
    const int    base[3] = { base0, base1, base2 };
    const int    so[3]   = { so0, so1, so2 };
    const double fr[3]   = { f0, f1, f2 };

    double wrap[3];
    double mod0 = 0.0;
    bool   valid = true;

    for (int i = 0; i < 3; ++i)
    {
        const int inum = sn[i];
        double sin_sum  = 0.0;
        double cos_sum  = 0.0;
        double sum_gray = 0.0;

        for (int k = 0; k < inum; ++k)
        {
            const double st = d_sin[so[i] + k];
            const double ct = d_cos[so[i] + k];
            const unsigned char* raw_frame = d_raw + (size_t)(base[i] + k) * frame_stride;
            const double pv = blurredSample(raw_frame, irow, jcol, rows, cols);
            sin_sum  -= st * pv;
            cos_sum  += ct * pv;
            sum_gray += pv;
        }

        wrap[i] = atan2(sin_sum, cos_sum);

        double b        = sin_sum * sin_sum + cos_sum * cos_sum;
        double mean_val = sum_gray / inum;
        b               = 2.0 * sqrt(b) / inum;

        double norm_mod = 0.0;
        if (mean_val > 1e-3)
            norm_mod = b / mean_val;
        if (i == 0)
            mod0 = norm_mod;

        if (norm_mod < mod_thresh)
        {
            valid = false;
            break;
        }
    }

    // Modulation map records frequency-0 value regardless of validity.
    d_mod[pix] = mod0;
    if (!valid)
        return;

    const double PI = 3.14159265358979323846;
    const double TWO_PI = 2.0 * PI;

    double phase12 = (wrap[0] > wrap[1]) ? (wrap[0] - wrap[1])
                                         : (wrap[0] - wrap[1] + TWO_PI);
    double phase23;
    if (fr[1] > fr[2])
        phase23 = (wrap[1] > wrap[2]) ? (wrap[1] - wrap[2])
                                     : (wrap[1] - wrap[2] + TWO_PI);
    else
        phase23 = (wrap[2] > wrap[1]) ? (wrap[2] - wrap[1])
                                     : (wrap[2] - wrap[1] + TWO_PI);
    double phase123 = (phase12 > phase23) ? (phase12 - phase23)
                                          : (phase12 - phase23 + TWO_PI);

    int f12 = abs((int)(fr[0] - fr[1]));
    int f23 = abs((int)(fr[1] - fr[2]));

    double uphase12 = phase12 + round((phase123 * f12 - phase12) / TWO_PI) * TWO_PI;

    double uphase1 = wrap[0] + round((uphase12 * fr[0] / f12 - wrap[0]) / TWO_PI) * TWO_PI;
    double k_val   = round((uphase12 * fr[0] / f12 - wrap[0]) / TWO_PI);

    double uphase23 = phase23 + round((phase123 * f23 - phase23) / TWO_PI) * TWO_PI;

    double uphase2 = wrap[1] + round((uphase23 * fr[1] / f23 - wrap[1]) / TWO_PI) * TWO_PI;
    uphase2 = (uphase2 * fr[0]) / fr[1];

    double uphase3 = wrap[2] + round((uphase23 * fr[2] / f23 - wrap[2]) / TWO_PI) * TWO_PI;
    uphase3 = (uphase3 * fr[0]) / fr[2];

    d_phase[pix] = avg_three ? (uphase1 + uphase2 + uphase3) / 3.0 : uphase1;
    d_k[pix]     = (int)k_val;
}

// ---------------------------------------------------------------------------
// Persistent device buffers, reused across decode calls (left / right frames
// of the same size).  Avoids re-allocating several GB of frame memory on every
// call.  Reclaimed by JSLAB_DecodePhase_GPU_Release() or at process exit.
// ---------------------------------------------------------------------------
static unsigned char* g_d_raw = nullptr;  static size_t g_raw_bytes   = 0;
static double* g_d_sin   = nullptr;  static size_t g_sin_bytes   = 0;
static double* g_d_cos   = nullptr;  static size_t g_cos_bytes   = 0;
static double* g_d_phase = nullptr;  static size_t g_phase_bytes = 0;
static double* g_d_mod   = nullptr;  static size_t g_mod_bytes   = 0;
static int*    g_d_k     = nullptr;  static size_t g_k_bytes     = 0;

static void ensureBuffer(void** ptr, size_t* cap, size_t need)
{
    if (*ptr != nullptr && *cap >= need)
        return;
    if (*ptr != nullptr)
        cudaFree(*ptr);
    cudaMalloc(ptr, need);
    *cap = need;
}

// Initializes the CUDA context (and persistent buffers lazily) so that the
// one-time driver/context setup is not charged to the first decode call.
void JSLAB_DecodePhase_GPU_Warmup()
{
    cudaFree(0);
}

void JSLAB_DecodePhase_GPU_Release()
{
    if (g_d_raw)   { cudaFree(g_d_raw);   g_d_raw   = nullptr; g_raw_bytes   = 0; }
    if (g_d_sin)   { cudaFree(g_d_sin);   g_d_sin   = nullptr; g_sin_bytes   = 0; }
    if (g_d_cos)   { cudaFree(g_d_cos);   g_d_cos   = nullptr; g_cos_bytes   = 0; }
    if (g_d_phase) { cudaFree(g_d_phase); g_d_phase = nullptr; g_phase_bytes = 0; }
    if (g_d_mod)   { cudaFree(g_d_mod);   g_d_mod   = nullptr; g_mod_bytes   = 0; }
    if (g_d_k)     { cudaFree(g_d_k);     g_d_k     = nullptr; g_k_bytes     = 0; }
}

// ---------------------------------------------------------------------------
// Host launcher: single upload of raw frames (as float), fused pre-blur +
// decode in one kernel, single download of phase / order / modulation maps.
// All device math is double precision to match the CPU decoder on unwrap order.
// ---------------------------------------------------------------------------
void JSLAB_DecodePhase_GPU(const std::vector<cv::Mat>& images,
                           const JSLAB_PhaseShiftDecoder::Config& cfg,
                           cv::Mat& phase, cv::Mat& k_map, cv::Mat& mod_map)
{
    const int sn0 = cfg.shift_num[0];
    const int sn1 = cfg.shift_num[1];
    const int sn2 = cfg.shift_num[2];
    const int total = sn0 + sn1 + sn2;

    if (total <= 0 || images.empty())
        return;

    const int rows = images[0].rows;
    const int cols = images[0].cols;
    const int frame_stride = rows * cols;
    const size_t raw_frame_bytes = (size_t)frame_stride * sizeof(unsigned char);
    const size_t out_bytes       = (size_t)frame_stride * sizeof(double);

    // ---- Pre-blur weights (matches cv::GaussianBlur(3x3, preblur_sigma)) ----
    cv::Mat k1d = cv::getGaussianKernel(3, cfg.preblur_sigma, CV_64F);
    cv::Mat k2d = k1d * k1d.t();
    double h_blur[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            h_blur[i * 3 + j] = k2d.at<double>(i, j);
    cudaMemcpyToSymbol(c_blur, h_blur, sizeof(h_blur));

    // ---- Sin/cos lookup tables (concatenated per frequency) ----
    std::vector<double> h_sin, h_cos;
    h_sin.reserve(total);
    h_cos.reserve(total);
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < 3; ++i)
    {
        const int inum = cfg.shift_num[i];
        for (int k = 0; k < inum; ++k)
        {
            h_sin.push_back(sin(2.0 * PI * k / inum));
            h_cos.push_back(cos(2.0 * PI * k / inum));
        }
    }

    // ---- Device buffers (persistent, reused across calls) ----
    ensureBuffer((void**)&g_d_raw,   &g_raw_bytes,   (size_t)total * raw_frame_bytes);
    ensureBuffer((void**)&g_d_sin,   &g_sin_bytes,   (size_t)total * sizeof(double));
    ensureBuffer((void**)&g_d_cos,   &g_cos_bytes,   (size_t)total * sizeof(double));
    ensureBuffer((void**)&g_d_phase, &g_phase_bytes, out_bytes);
    ensureBuffer((void**)&g_d_mod,   &g_mod_bytes,   out_bytes);
    ensureBuffer((void**)&g_d_k,     &g_k_bytes,     (size_t)frame_stride * sizeof(int));

    cudaMemset(g_d_phase, 0, out_bytes);
    cudaMemset(g_d_mod,   0, out_bytes);
    cudaMemset(g_d_k,     0, (size_t)frame_stride * sizeof(int));

    cudaMemcpy(g_d_sin, h_sin.data(), (size_t)total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_cos, h_cos.data(), (size_t)total * sizeof(double), cudaMemcpyHostToDevice);

    // ---- Upload frames as 8-bit grayscale (4x less PCIe / device memory than
    // float).  Structured-light captures are 8-bit; non-CV_8U inputs are
    // converted to CV_8U for upload. ----
    for (int idx = 0; idx < total && idx < (int)images.size(); ++idx)
    {
        cv::Mat u8;
        if (images[idx].type() == CV_8UC1)
            u8 = images[idx];
        else
            images[idx].convertTo(u8, CV_8U);
        if (!u8.isContinuous())
            u8 = u8.clone();
        cudaMemcpy(g_d_raw + (size_t)idx * frame_stride, u8.ptr<unsigned char>(),
                   raw_frame_bytes, cudaMemcpyHostToDevice);
    }

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    // ---- Fused pre-blur + decode ----
    decodeKernel<<<grid, block>>>(
        g_d_raw, rows, cols,
        g_d_sin, g_d_cos,
        sn0, sn1, sn2,
        0, sn0, sn0 + sn1,
        0, sn0, sn0 + sn1,
        cfg.freq[0], cfg.freq[1], cfg.freq[2],
        cfg.modulation_threshold, cfg.avg_three_phase ? 1 : 0,
        g_d_phase, g_d_k, g_d_mod);

    cudaDeviceSynchronize();

    // ---- Download directly into the public CV_64F / CV_32S layout ----
    phase.create(rows, cols, CV_64FC1);
    mod_map.create(rows, cols, CV_64FC1);
    k_map.create(rows, cols, CV_32SC1);

    cudaMemcpy(phase.ptr<double>(),   g_d_phase, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(mod_map.ptr<double>(), g_d_mod,   out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(k_map.ptr<int>(),      g_d_k,     (size_t)frame_stride * sizeof(int),
               cudaMemcpyDeviceToHost);
}

#endif // JSLAB_USE_CUDA
