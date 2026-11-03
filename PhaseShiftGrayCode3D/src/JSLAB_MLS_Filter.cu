#include "JSLAB_Config.h"
#ifdef JSLAB_USE_CUDA

#include "JSLAB_MLS_Filter.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ---------------------------------------------------------------------------
// Device helper: analytical eigendecomposition of a 3x3 real symmetric matrix.
// Returns eigenvalues in ascending order; evecs[i] is the eigenvector for
// evals[i].  Uses Cardano / trigonometric solution for the cubic.
// ---------------------------------------------------------------------------
__device__ static void symmetricEigen3x3_device(
    const float C[3][3], float evals[3], float evecs[3][3])
{
    const float p1 = C[0][1] * C[0][1] + C[0][2] * C[0][2] + C[1][2] * C[1][2];

    if (p1 < 1e-20f)
    {
        evals[0] = C[0][0]; evals[1] = C[1][1]; evals[2] = C[2][2];
        evecs[0][0] = 1; evecs[0][1] = 0; evecs[0][2] = 0;
        evecs[1][0] = 0; evecs[1][1] = 1; evecs[1][2] = 0;
        evecs[2][0] = 0; evecs[2][1] = 0; evecs[2][2] = 1;

        for (int i = 0; i < 2; i++)
            for (int j = i + 1; j < 3; j++)
                if (evals[i] > evals[j])
                {
                    float tmp = evals[i]; evals[i] = evals[j]; evals[j] = tmp;
                    for (int k = 0; k < 3; k++)
                    {
                        tmp = evecs[i][k]; evecs[i][k] = evecs[j][k]; evecs[j][k] = tmp;
                    }
                }
        return;
    }

    const float tr = C[0][0] + C[1][1] + C[2][2];
    const float q  = tr / 3.0f;

    float B[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            B[i][j] = C[i][j];
    B[0][0] -= q; B[1][1] -= q; B[2][2] -= q;

    float p2 = B[0][0] * B[0][0] + B[1][1] * B[1][1] + B[2][2] * B[2][2]
             + 2.0f * p1;
    float p     = sqrtf(p2 / 6.0f);
    float inv_p = (p > 1e-15f) ? (1.0f / p) : 0.0f;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            B[i][j] *= inv_p;

    float detB = B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1])
               - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
               + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]);
    float r = detB * 0.5f;
    r = fmaxf(-1.0f, fminf(1.0f, r));

    float phi = acosf(r) / 3.0f;
    const float PI = 3.14159265358979323846f;

    evals[0] = q + 2.0f * p * cosf(phi + 2.0f * PI / 3.0f);
    evals[2] = q + 2.0f * p * cosf(phi);
    evals[1] = tr - evals[0] - evals[2];

    for (int idx = 0; idx < 3; idx++)
    {
        float A[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                A[i][j] = C[i][j];
        A[0][0] -= evals[idx];
        A[1][1] -= evals[idx];
        A[2][2] -= evals[idx];

        float c01[3] = {
            A[0][1] * A[1][2] - A[0][2] * A[1][1],
            A[0][2] * A[1][0] - A[0][0] * A[1][2],
            A[0][0] * A[1][1] - A[0][1] * A[1][0]
        };
        float c02[3] = {
            A[0][1] * A[2][2] - A[0][2] * A[2][1],
            A[0][2] * A[2][0] - A[0][0] * A[2][2],
            A[0][0] * A[2][1] - A[0][1] * A[2][0]
        };
        float c12[3] = {
            A[1][1] * A[2][2] - A[1][2] * A[2][1],
            A[1][2] * A[2][0] - A[1][0] * A[2][2],
            A[1][0] * A[2][1] - A[1][1] * A[2][0]
        };

        float len01 = c01[0] * c01[0] + c01[1] * c01[1] + c01[2] * c01[2];
        float len02 = c02[0] * c02[0] + c02[1] * c02[1] + c02[2] * c02[2];
        float len12 = c12[0] * c12[0] + c12[1] * c12[1] + c12[2] * c12[2];

        float* best     = c12;
        float  best_len = len12;
        if (len01 > best_len) { best = c01; best_len = len01; }
        if (len02 > best_len) { best = c02; best_len = len02; }

        float inv_len = (best_len > 1e-20f) ? rsqrtf(best_len) : 0.0f;
        evecs[idx][0] = best[0] * inv_len;
        evecs[idx][1] = best[1] * inv_len;
        evecs[idx][2] = best[2] * inv_len;
    }
}

// ---------------------------------------------------------------------------
// Device Cholesky solver for nc x nc (nc <= 6) SPD system.
// Returns false when the matrix is not positive-definite.
// ---------------------------------------------------------------------------
__device__ static bool choleskySolve_device(
    const float A[][6], const float b[], float x[], int nc)
{
    float L[6][6] = {};

    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            float s = A[i][j];
            for (int k = 0; k < j; k++)
                s -= L[i][k] * L[j][k];

            if (i == j)
            {
                if (s <= 1e-12f)
                    return false;
                L[i][j] = sqrtf(s);
            }
            else
            {
                L[i][j] = s / L[j][j];
            }
        }
    }

    float y[6] = {};
    for (int i = 0; i < nc; i++)
    {
        float s = b[i];
        for (int k = 0; k < i; k++)
            s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }

    for (int i = nc - 1; i >= 0; i--)
    {
        float s = y[i];
        for (int k = i + 1; k < nc; k++)
            s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }

    return true;
}

// ---------------------------------------------------------------------------
// CUDA kernel: one thread per pixel – supports polynomial_order 0 / 1 / 2
// ---------------------------------------------------------------------------
__global__ void mlsKernel(
    const float* __restrict__ d_xyz,
    const unsigned char* __restrict__ d_P,
    float* __restrict__ d_out,
    int rows, int cols,
    int hw,
    float inv_2sigma2,
    int min_neighbors_0,
    int polynomial_order,
    float radius2_cutoff)
{
    int jcol = blockIdx.x * blockDim.x + threadIdx.x;
    int irow = blockIdx.y * blockDim.y + threadIdx.y;
    if (irow >= rows || jcol >= cols)
        return;

    int idx = irow * cols + jcol;

    d_out[idx * 3 + 0] = d_xyz[idx * 3 + 0];
    d_out[idx * 3 + 1] = d_xyz[idx * 3 + 1];
    d_out[idx * 3 + 2] = d_xyz[idx * 3 + 2];

    if (d_P[idx] == 0)
        return;

    float px = d_xyz[idx * 3 + 0];
    float py = d_xyz[idx * 3 + 1];
    float pz = d_xyz[idx * 3 + 2];

    const int min_nb[3] = { 3, 5, 8 };

    // ---- 1. Weighted centroid ----
    float cx = 0, cy = 0, cz = 0, w_sum = 0;
    int n_valid = 0;

    int r0 = max(0, irow - hw);
    int r1 = min(rows, irow + hw + 1);
    int c0 = max(0, jcol - hw);
    int c1 = min(cols, jcol + hw + 1);

    for (int ir = r0; ir < r1; ir++)
    {
        for (int jc = c0; jc < c1; jc++)
        {
            int kid = ir * cols + jc;
            if (d_P[kid] == 0)
                continue;

            float kx = d_xyz[kid * 3 + 0];
            float ky = d_xyz[kid * 3 + 1];
            float kz = d_xyz[kid * 3 + 2];

            float dx = kx - px, dy = ky - py, dz = kz - pz;
            float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 > radius2_cutoff)
                continue;
            float w = expf(-dist2 * inv_2sigma2);

            cx += w * kx;
            cy += w * ky;
            cz += w * kz;
            w_sum += w;
            n_valid++;
        }
    }

    if (n_valid < min_nb[0] || w_sum < 1e-12f)
        return;

    float inv_w = 1.0f / w_sum;
    cx *= inv_w;
    cy *= inv_w;
    cz *= inv_w;

    // ---- 2. Weighted covariance ----
    float cov[3][3] = {};

    for (int ir = r0; ir < r1; ir++)
    {
        for (int jc = c0; jc < c1; jc++)
        {
            int kid = ir * cols + jc;
            if (d_P[kid] == 0)
                continue;

            float kx = d_xyz[kid * 3 + 0];
            float ky = d_xyz[kid * 3 + 1];
            float kz = d_xyz[kid * 3 + 2];

            float dx = kx - px, dy = ky - py, dz = kz - pz;
            float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 > radius2_cutoff)
                continue;
            float w = expf(-dist2 * inv_2sigma2);

            float ex = kx - cx, ey = ky - cy, ez = kz - cz;
            cov[0][0] += w * ex * ex;
            cov[0][1] += w * ex * ey;
            cov[0][2] += w * ex * ez;
            cov[1][1] += w * ey * ey;
            cov[1][2] += w * ey * ez;
            cov[2][2] += w * ez * ez;
        }
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];

    // ---- 3. Eigendecomposition – local frame ----
    float evals[3];
    float evecs[3][3];
    symmetricEigen3x3_device(cov, evals, evecs);

    float nx  = evecs[0][0], ny  = evecs[0][1], nz  = evecs[0][2];
    float t1x = evecs[1][0], t1y = evecs[1][1], t1z = evecs[1][2];
    float t2x = evecs[2][0], t2y = evecs[2][1], t2z = evecs[2][2];

    // ---- 4. Determine effective order (auto fallback) ----
    int cur_order = polynomial_order;
    if (cur_order > 2) cur_order = 2;
    if (cur_order > 0 && n_valid < min_nb[cur_order])
        cur_order = (n_valid >= min_nb[1]) ? 1 : 0;

    if (cur_order == 0)
    {
        float dot = (px - cx) * nx + (py - cy) * ny + (pz - cz) * nz;
        d_out[idx * 3 + 0] = px - dot * nx;
        d_out[idx * 3 + 1] = py - dot * ny;
        d_out[idx * 3 + 2] = pz - dot * nz;
        return;
    }

    // ---- 5. Build weighted normal equations ----
    const int nc = (cur_order == 1) ? 3 : 6;

    float AtWA[6][6] = {};
    float AtWf[6]    = {};

    for (int ir = r0; ir < r1; ir++)
    {
        for (int jc = c0; jc < c1; jc++)
        {
            int kid = ir * cols + jc;
            if (d_P[kid] == 0)
                continue;

            float kx = d_xyz[kid * 3 + 0];
            float ky = d_xyz[kid * 3 + 1];
            float kz = d_xyz[kid * 3 + 2];

            float dx = kx - px, dy = ky - py, dz = kz - pz;
            float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 > radius2_cutoff)
                continue;
            float w = expf(-dist2 * inv_2sigma2);

            float ex = kx - cx, ey = ky - cy, ez = kz - cz;
            float u = ex * t1x + ey * t1y + ez * t1z;
            float v = ex * t2x + ey * t2y + ez * t2z;
            float f = ex * nx  + ey * ny  + ez * nz;

            float basis[6];
            basis[0] = 1.0f;
            basis[1] = u;
            basis[2] = v;
            if (nc == 6)
            {
                basis[3] = u * u;
                basis[4] = u * v;
                basis[5] = v * v;
            }

            for (int a = 0; a < nc; a++)
            {
                AtWf[a] += w * basis[a] * f;
                for (int b = a; b < nc; b++)
                    AtWA[a][b] += w * basis[a] * basis[b];
            }
        }
    }

    for (int a = 0; a < nc; a++)
        for (int b = 0; b < a; b++)
            AtWA[a][b] = AtWA[b][a];

    // ---- 6. Solve – with fallback ----
    float coeffs[6] = {};
    bool solved = choleskySolve_device(AtWA, AtWf, coeffs, nc);

    if (!solved && cur_order == 2)
    {
        float AtWA1[6][6] = {};
        float AtWf1[6]    = {};
        for (int a = 0; a < 3; a++)
        {
            AtWf1[a] = AtWf[a];
            for (int b = 0; b < 3; b++)
                AtWA1[a][b] = AtWA[a][b];
        }
        solved = choleskySolve_device(AtWA1, AtWf1, coeffs, 3);
        if (solved)
            cur_order = 1;
    }

    if (!solved)
    {
        float dot = (px - cx) * nx + (py - cy) * ny + (pz - cz) * nz;
        d_out[idx * 3 + 0] = px - dot * nx;
        d_out[idx * 3 + 1] = py - dot * ny;
        d_out[idx * 3 + 2] = pz - dot * nz;
        return;
    }

    // ---- 7. Evaluate polynomial at query point ----
    float ex0 = px - cx, ey0 = py - cy, ez0 = pz - cz;
    float u0 = ex0 * t1x + ey0 * t1y + ez0 * t1z;
    float v0 = ex0 * t2x + ey0 * t2y + ez0 * t2z;

    float f_fitted = coeffs[0] + coeffs[1] * u0 + coeffs[2] * v0;
    if (cur_order == 2)
        f_fitted += coeffs[3] * u0 * u0 + coeffs[4] * u0 * v0 + coeffs[5] * v0 * v0;

    // ---- 8. Transform back to world coordinates ----
    d_out[idx * 3 + 0] = cx + u0 * t1x + v0 * t2x + f_fitted * nx;
    d_out[idx * 3 + 1] = cy + u0 * t1y + v0 * t2y + f_fitted * ny;
    d_out[idx * 3 + 2] = cz + u0 * t1z + v0 * t2z + f_fitted * nz;
}

// ---------------------------------------------------------------------------
// Host wrapper – supports multi-iteration via double-buffering on device
// ---------------------------------------------------------------------------
void applyMLSFilterToPointCloud_GPU(
    const cv::Mat& src_xyz, const cv::Mat& P,
    cv::Mat& dst_xyz, int window_size, float spatial_sigma,
    int polynomial_order, float max_radius, int iterations)
{
    const int rows = src_xyz.rows;
    const int cols = src_xyz.cols;
    const int n_pixels = rows * cols;
    const float inv_2sigma2 = 1.0f / (2.0f * spatial_sigma * spatial_sigma);
    const float radius2_cutoff = (max_radius > 0.0f)
        ? max_radius * max_radius
        : 9.0f * spatial_sigma * spatial_sigma;
    const int min_neighbors = 3;
    const int n_iter = (iterations > 1) ? iterations : 1;

    cv::Mat src_cont;
    if (src_xyz.isContinuous())
        src_cont = src_xyz;
    else
        src_cont = src_xyz.clone();

    cv::Mat P_cont;
    if (P.isContinuous())
        P_cont = P;
    else
        P_cont = P.clone();

    float*         d_buf0 = nullptr;
    float*         d_buf1 = nullptr;
    unsigned char* d_P    = nullptr;

    size_t xyz_bytes = (size_t)n_pixels * 3 * sizeof(float);
    size_t P_bytes   = (size_t)n_pixels * sizeof(unsigned char);

    cudaMalloc(&d_buf0, xyz_bytes);
    cudaMalloc(&d_buf1, xyz_bytes);
    cudaMalloc(&d_P,    P_bytes);

    cudaMemcpy(d_buf0, src_cont.ptr<float>(),       xyz_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P,    P_cont.ptr<unsigned char>(),  P_bytes,   cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    float* d_in  = d_buf0;
    float* d_out = d_buf1;

    for (int iter = 0; iter < n_iter; iter++)
    {
        mlsKernel<<<grid, block>>>(d_in, d_P, d_out,
                                    rows, cols, window_size,
                                    inv_2sigma2, min_neighbors,
                                    polynomial_order, radius2_cutoff);
        cudaDeviceSynchronize();

        if (iter + 1 < n_iter)
        {
            float* tmp = d_in;
            d_in  = d_out;
            d_out = tmp;
        }
    }

    dst_xyz.create(rows, cols, CV_32FC3);
    cudaMemcpy(dst_xyz.ptr<float>(), d_out, xyz_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_buf0);
    cudaFree(d_buf1);
    cudaFree(d_P);
}

#endif // JSLAB_USE_CUDA
