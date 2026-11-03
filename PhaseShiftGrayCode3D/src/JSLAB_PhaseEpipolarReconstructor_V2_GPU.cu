#include "JSLAB_PhaseEpipolarReconstructor_V2_GPU.h"
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

// 3x3 point-cloud Gaussian weights (outer product of getGaussianKernel(3, 0.7)).
__constant__ float c_pc_gauss[9];

// ---------------------------------------------------------------------------
// Helper for Module A: per-row column range of each fringe order in the right
// image.  One thread per right pixel records, for its order k, the min and max
// column where that order occurs (atomicMin/atomicMax into a (knum+1)-bin table
// per row).  This lets the stereo match scan only the contiguous candidate
// span instead of the whole row, while staying bit-exact (we still re-check
// r_k == k inside the span).
// ---------------------------------------------------------------------------
__global__ void buildOrderRangeKernel(
    const int* __restrict__ r_k, int rows, int cols, int knum,
    int* __restrict__ col_min, int* __restrict__ col_max)
{
    int jR   = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || jR >= cols)
        return;

    int k = r_k[irow * cols + jR];
    if (k >= 0 && k <= knum)
    {
        int base = irow * (knum + 1) + k;
        atomicMin(&col_min[base], jR);
        atomicMax(&col_max[base], jR);
    }
}

// ---------------------------------------------------------------------------
// Module A: order-constrained stereo matching with sub-pixel interpolation.
//
// One thread per left pixel.  Replaces the CPU std::vector row_rank_table with
// a precomputed per-row, per-order column span (col_min/col_max).  For a left
// pixel with order k, only the right columns in [col_min, col_max] are scanned
// (still filtered by r_k == k), which reproduces the CPU result exactly while
// avoiding a full-row O(cols) scan per pixel.
// ---------------------------------------------------------------------------
__global__ void stereoMatchKernel(
    const double* __restrict__ l_phase, const double* __restrict__ r_phase,
    const int* __restrict__ l_k, const int* __restrict__ r_k,
    const int* __restrict__ col_min, const int* __restrict__ col_max,
    int rows, int cols,
    int pha_f, double grad_threshold,
    float* __restrict__ disp)
{
    int j    = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || j >= cols)
        return;

    const int row_off = irow * cols;
    disp[row_off + j] = 0.0f;

    const int max_rank_limit = pha_f;
    const int knum           = pha_f;

    int k = l_k[row_off + j];
    if (k > max_rank_limit || k <= 0)
        return;
    if (k > knum)
        return;

    // Candidate column span for this order in the right row.
    const int span_base = irow * (knum + 1) + k;
    const int jR_lo = col_min[span_base];
    const int jR_hi = col_max[span_base];
    if (jR_lo > jR_hi)   // no right pixel has this order in this row
        return;

    // High-frequency absolute phases are large; keep the residual / parabola
    // math in double to avoid the single-precision truncation that corrupted
    // diff and delta_L.
    const double l_val = l_phase[row_off + j];
    double min_cost = 1e9;
    int    best_jR  = -1;

    for (int jR = jR_lo; jR <= jR_hi; ++jR)
    {
        if (r_k[row_off + jR] != k)
            continue;
        double diff = fabs(r_phase[row_off + jR] - l_val);
        if (diff < min_cost)
        {
            min_cost = diff;
            best_jR  = jR;
        }
    }

    if (best_jR < 0 || min_cost >= grad_threshold)
        return;
    if (best_jR < 2 || best_jR >= cols - 1)
        return;

    double cost0 = r_phase[row_off + best_jR - 1] - l_val;
    double cost1 = r_phase[row_off + best_jR]     - l_val;
    double cost2 = r_phase[row_off + best_jR + 1] - l_val;

    double numeratorL   = cost1 * (cost0 - cost2);
    double denominatorL = 2.0 * (cost0 - cost1) * (cost1 - cost2);

    double delta_L = 0.0;
    if (fabs(denominatorL) > 1e-9)
        delta_L = numeratorL / denominatorL;

    disp[row_off + j] = (float)((double)j - (double)best_jR - delta_L);
}

// ---------------------------------------------------------------------------
// Optional 5x5 median filter on the disparity map (BORDER_REPLICATE), matching
// cv::medianBlur applied in the CPU pipeline.
// ---------------------------------------------------------------------------
__global__ void medianFilter5x5Kernel(
    const float* __restrict__ src, float* __restrict__ dst,
    int rows, int cols)
{
    int jcol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || jcol >= cols)
        return;

    float v[25];
    int n = 0;
    for (int dy = -2; dy <= 2; ++dy)
    {
        int yy = min(max(irow + dy, 0), rows - 1);
        for (int dx = -2; dx <= 2; ++dx)
        {
            int xx = min(max(jcol + dx, 0), cols - 1);
            v[n++] = src[yy * cols + xx];
        }
    }

    // Partial selection up to the median index (13th element, index 12).
    for (int i = 0; i <= 12; ++i)
    {
        int mi = i;
        for (int t = i + 1; t < 25; ++t)
            if (v[t] < v[mi])
                mi = t;
        float tmp = v[i]; v[i] = v[mi]; v[mi] = tmp;
    }
    dst[irow * cols + jcol] = v[12];
}

// ---------------------------------------------------------------------------
// Module B: reproject disparity to 3-D.  Q perspective divide and R1 rotation
// use double; depth clamp + use_rectified_coords branch as in the CPU version.
// Outputs xyz (float3, flattened) and the validity mask P.
// ---------------------------------------------------------------------------
__global__ void reprojectTo3DKernel(
    const float* __restrict__ disp,
    int rows, int cols,
    const double* __restrict__ Q, const double* __restrict__ R1,
    double min_depth, double max_depth, int use_rectified,
    float* __restrict__ xyz, unsigned char* __restrict__ P)
{
    int jcol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || jcol >= cols)
        return;

    const int pix = irow * cols + jcol;
    xyz[pix * 3 + 0] = 0.0f;
    xyz[pix * 3 + 1] = 0.0f;
    xyz[pix * 3 + 2] = 0.0f;
    P[pix] = 0;

    double d = (double)disp[pix];
    if (fabs(d) < 1e-5)
        return;

    double w = Q[14] * d + Q[15];
    double x = (double)jcol + Q[3];
    double y = (double)irow + Q[7];
    double z = Q[11];

    double inv_w = 1.0 / w;
    x *= inv_w;
    y *= inv_w;
    z *= inv_w;

    double rx = R1[0] * x + R1[3] * y + R1[6] * z;
    double ry = R1[1] * x + R1[4] * y + R1[7] * z;
    double rz = R1[2] * x + R1[5] * y + R1[8] * z;

    if (use_rectified)
    {
        if (z > min_depth && z < max_depth)
        {
            xyz[pix * 3 + 0] = (float)x;
            xyz[pix * 3 + 1] = (float)y;
            xyz[pix * 3 + 2] = (float)z;
            P[pix] = 1;
        }
    }
    else
    {
        if (rz > min_depth && rz < max_depth)
        {
            xyz[pix * 3 + 0] = (float)rx;
            xyz[pix * 3 + 1] = (float)ry;
            xyz[pix * 3 + 2] = (float)rz;
            P[pix] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Module C1: point-cloud Gaussian filter (3x3 window).  Preserves the
// non-standard X-coordinate truncation (|x_neighbor - x_center| < 3.0) and the
// count > 0.8 acceptance threshold from the CPU implementation.
// ---------------------------------------------------------------------------
__global__ void gaussianFilterKernel(
    const float* __restrict__ src_xyz, const unsigned char* __restrict__ P,
    float* __restrict__ dst_xyz, unsigned char* __restrict__ dst_P,
    int rows, int cols)
{
    int jcol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || jcol >= cols)
        return;

    const int pix = irow * cols + jcol;
    dst_xyz[pix * 3 + 0] = 0.0f;
    dst_xyz[pix * 3 + 1] = 0.0f;
    dst_xyz[pix * 3 + 2] = 0.0f;
    dst_P[pix] = 0;

    const int half = 1;  // win_size = 3
    if (irow < half || irow >= rows - half || jcol < half || jcol >= cols - half)
        return;
    if (P[pix] == 0)
        return;

    const float count_threshold = 0.8f;
    const float cx = src_xyz[pix * 3 + 0];

    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    float count = 0.0f;

    for (int ik = -half; ik <= half; ++ik)
    {
        for (int jk = -half; jk <= half; ++jk)
        {
            int nb = (irow + ik) * cols + (jcol + jk);
            if (P[nb] == 0)
                continue;
            float nx = src_xyz[nb * 3 + 0];
            if (fabsf(nx - cx) < 3.0f)
            {
                float wgt = c_pc_gauss[(ik + half) * 3 + (jk + half)];
                sx += src_xyz[nb * 3 + 0] * wgt;
                sy += src_xyz[nb * 3 + 1] * wgt;
                sz += src_xyz[nb * 3 + 2] * wgt;
                count += wgt;
            }
        }
    }

    if (count > count_threshold)
    {
        dst_xyz[pix * 3 + 0] = sx / count;
        dst_xyz[pix * 3 + 1] = sy / count;
        dst_xyz[pix * 3 + 2] = sz / count;
        dst_P[pix] = 1;
    }
}

// ---------------------------------------------------------------------------
// Module C2: radius outlier filter.  One thread per pixel; counts neighbours
// within the 3-D search radius inside a (2*half+1) pixel window and keeps the
// point when the count reaches min_neighbors.
// ---------------------------------------------------------------------------
__global__ void radiusOutlierKernel(
    const float* __restrict__ src_xyz, const unsigned char* __restrict__ P,
    float* __restrict__ dst_xyz, unsigned char* __restrict__ dst_P,
    int rows, int cols,
    float radius_squared, int min_neighbors, int half_win_size)
{
    int jcol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || jcol >= cols)
        return;

    const int pix = irow * cols + jcol;
    dst_xyz[pix * 3 + 0] = 0.0f;
    dst_xyz[pix * 3 + 1] = 0.0f;
    dst_xyz[pix * 3 + 2] = 0.0f;
    dst_P[pix] = 0;

    if (P[pix] == 0)
        return;

    const float px = src_xyz[pix * 3 + 0];
    const float py = src_xyz[pix * 3 + 1];
    const float pz = src_xyz[pix * 3 + 2];

    int row_start = max(0, irow - half_win_size);
    int row_end   = min(rows, irow + half_win_size + 1);
    int col_start = max(0, jcol - half_win_size);
    int col_end   = min(cols, jcol + half_win_size + 1);

    int neighbor_count = 0;
    bool keep = false;

    for (int i = row_start; i < row_end && !keep; ++i)
    {
        for (int jj = col_start; jj < col_end; ++jj)
        {
            int nb = i * cols + jj;
            if (P[nb] == 0)
                continue;
            if (i == irow && jj == jcol)
                continue;

            float dx = px - src_xyz[nb * 3 + 0];
            float dy = py - src_xyz[nb * 3 + 1];
            float dz = pz - src_xyz[nb * 3 + 2];
            float dist2 = dx * dx + dy * dy + dz * dz;

            if (dist2 <= radius_squared)
            {
                ++neighbor_count;
                if (neighbor_count >= min_neighbors)
                {
                    keep = true;
                    break;
                }
            }
        }
    }

    if (keep)
    {
        dst_xyz[pix * 3 + 0] = px;
        dst_xyz[pix * 3 + 1] = py;
        dst_xyz[pix * 3 + 2] = pz;
        dst_P[pix] = 1;
    }
}

// ---------------------------------------------------------------------------
// Host launcher: single upload, whole pipeline device-to-device, single
// download.  MLS smoothing is applied by the caller (reuses the existing GPU
// MLS launcher) so this stays focused on the matching/reproject/filter stages.
// ---------------------------------------------------------------------------
void JSLAB_ReconstructPipeline_GPU(
    const cv::Mat& l_phase, const cv::Mat& r_phase,
    const cv::Mat& l_k, const cv::Mat& r_k,
    int pha_f, double min_disp, double max_disp,
    const double Q[16], const double R1[9],
    const JSLAB_PhaseEpipolarReconstructor_V2_GPU::Config& cfg,
    cv::Mat& xyz, cv::Mat& P, cv::Mat& disp_map)
{
    const int rows = l_phase.rows;
    const int cols = l_phase.cols;
    const int n_pixels = rows * cols;
    if (n_pixels <= 0)
        return;

    // The CPU stereo match scans the whole row (not disparity-bounded); the
    // estimated disparity range is intentionally unused here to keep results
    // identical to the CPU pipeline.
    (void)min_disp;
    (void)max_disp;

    // ---- Host-side type normalization ----
    // Keep the phase maps in double: the absolute phase values are large, and
    // converting to float truncates the residual / parabola math in the stereo
    // match (wrong delta_L, corrupt disparity).
    cv::Mat lph64, rph64;
    if (l_phase.type() == CV_64FC1) lph64 = l_phase;
    else                            l_phase.convertTo(lph64, CV_64F);
    if (r_phase.type() == CV_64FC1) rph64 = r_phase;
    else                            r_phase.convertTo(rph64, CV_64F);
    if (!lph64.isContinuous()) lph64 = lph64.clone();
    if (!rph64.isContinuous()) rph64 = rph64.clone();

    cv::Mat lk32, rk32;
    if (l_k.type() == CV_32SC1) lk32 = l_k.isContinuous() ? l_k : l_k.clone();
    else                        l_k.convertTo(lk32, CV_32S);
    if (r_k.type() == CV_32SC1) rk32 = r_k.isContinuous() ? r_k : r_k.clone();
    else                        r_k.convertTo(rk32, CV_32S);

    // ---- Point-cloud Gaussian weights ----
    cv::Mat g1d = cv::getGaussianKernel(3, 0.7, CV_32F);
    cv::Mat g2d = g1d * g1d.t();
    float h_gauss[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            h_gauss[i * 3 + j] = g2d.at<float>(i, j);
    cudaMemcpyToSymbol(c_pc_gauss, h_gauss, sizeof(h_gauss));

    // ---- Device allocations ----
    const size_t d_bytes   = (size_t)n_pixels * sizeof(double);
    const size_t f_bytes   = (size_t)n_pixels * sizeof(float);
    const size_t i_bytes   = (size_t)n_pixels * sizeof(int);
    const size_t xyz_bytes = (size_t)n_pixels * 3 * sizeof(float);
    const size_t p_bytes   = (size_t)n_pixels * sizeof(unsigned char);

    double* d_lph  = nullptr; double* d_rph  = nullptr;
    int*   d_lk    = nullptr; int*   d_rk    = nullptr;
    float* d_disp  = nullptr; float* d_disp2 = nullptr;
    float* d_xyz   = nullptr; float* d_xyz_g = nullptr; float* d_xyz_r = nullptr;
    unsigned char* d_P = nullptr; unsigned char* d_Pg = nullptr; unsigned char* d_Pr = nullptr;
    double* d_Q = nullptr; double* d_R1 = nullptr;

    // Per-row, per-order column span table for the bounded stereo match.
    const int knum = (pha_f > 0) ? pha_f : 0;
    const size_t span_count = (size_t)rows * (knum + 1);
    const size_t span_bytes = span_count * sizeof(int);
    int* d_col_min = nullptr; int* d_col_max = nullptr;

    cudaMalloc(&d_lph, d_bytes);   cudaMalloc(&d_rph, d_bytes);
    cudaMalloc(&d_lk,  i_bytes);   cudaMalloc(&d_rk,  i_bytes);
    cudaMalloc(&d_disp, f_bytes);  cudaMalloc(&d_disp2, f_bytes);
    cudaMalloc(&d_xyz, xyz_bytes); cudaMalloc(&d_xyz_g, xyz_bytes);
    cudaMalloc(&d_xyz_r, xyz_bytes);
    cudaMalloc(&d_P, p_bytes);     cudaMalloc(&d_Pg, p_bytes);     cudaMalloc(&d_Pr, p_bytes);
    cudaMalloc(&d_Q, 16 * sizeof(double));
    cudaMalloc(&d_R1, 9 * sizeof(double));
    cudaMalloc(&d_col_min, span_bytes);
    cudaMalloc(&d_col_max, span_bytes);

    // cudaMalloc does NOT zero memory: clear the disparity buffers so that any
    // pixel not written by the stereo-match kernel stays 0 instead of leaving
    // garbage that would crash the reproject stage.
    cudaMemset(d_disp,  0, f_bytes);
    cudaMemset(d_disp2, 0, f_bytes);

    // Initialize the span table: col_min -> large positive (0x7F7F7F7F),
    // col_max -> -1 (0xFFFFFFFF), so atomicMin/atomicMax start from extremes
    // and an order with no pixels keeps jR_lo > jR_hi.
    cudaMemset(d_col_min, 0x7F, span_bytes);
    cudaMemset(d_col_max, 0xFF, span_bytes);

    cudaMemcpy(d_lph, lph64.ptr<double>(), d_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rph, rph64.ptr<double>(), d_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lk,  lk32.ptr<int>(),    i_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rk,  rk32.ptr<int>(),    i_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q,   Q,  16 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R1,  R1,  9 * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    // ---- Module A: build per-row order spans, then bounded stereo match ----
    buildOrderRangeKernel<<<grid, block>>>(d_rk, rows, cols, knum, d_col_min, d_col_max);
    stereoMatchKernel<<<grid, block>>>(
        d_lph, d_rph, d_lk, d_rk, d_col_min, d_col_max, rows, cols,
        pha_f, cfg.grad_threshold, d_disp);

    // ---- Optional 5x5 median on disparity ----
    float* disp_in = d_disp;
    if (cfg.use_disp_median_filter)
    {
        medianFilter5x5Kernel<<<grid, block>>>(d_disp, d_disp2, rows, cols);
        disp_in = d_disp2;
    }

    // ---- Module B: reproject to 3-D ----
    reprojectTo3DKernel<<<grid, block>>>(
        disp_in, rows, cols, d_Q, d_R1,
        cfg.min_depth, cfg.max_depth, cfg.use_rectified_coords ? 1 : 0,
        d_xyz, d_P);

    // ---- Module C1: point-cloud Gaussian filter ----
    // Writes an updated mask d_Pg: points discarded by the count threshold are
    // marked invalid so they cannot collapse onto the origin and survive ROR.
    gaussianFilterKernel<<<grid, block>>>(d_xyz, d_P, d_xyz_g, d_Pg, rows, cols);

    // ---- Module C2: radius outlier filter (uses the Gaussian-updated mask) ----
    const float radius_squared = cfg.radius_search * cfg.radius_search;
    const int   half_win       = cfg.radius_window_size / 2;
    radiusOutlierKernel<<<grid, block>>>(
        d_xyz_g, d_Pg, d_xyz_r, d_Pr, rows, cols,
        radius_squared, cfg.radius_min_neighbors, half_win);

    cudaDeviceSynchronize();

    // ---- Download filtered point cloud + mask ----
    xyz.create(rows, cols, CV_32FC3);
    P.create(rows, cols, CV_8UC1);
    cudaMemcpy(xyz.ptr<float>(),         d_xyz_r, xyz_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(P.ptr<unsigned char>(),   d_Pr,    p_bytes,   cudaMemcpyDeviceToHost);

    // ---- [Diagnostic] download the disparity (after stereo match + median) ----
    disp_map.create(rows, cols, CV_32FC1);
    cudaMemcpy(disp_map.ptr<float>(), disp_in, f_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_lph);   cudaFree(d_rph);
    cudaFree(d_lk);    cudaFree(d_rk);
    cudaFree(d_disp);  cudaFree(d_disp2);
    cudaFree(d_xyz);   cudaFree(d_xyz_g); cudaFree(d_xyz_r);
    cudaFree(d_P);     cudaFree(d_Pg);    cudaFree(d_Pr);
    cudaFree(d_Q);     cudaFree(d_R1);
    cudaFree(d_col_min); cudaFree(d_col_max);
}
