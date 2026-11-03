#include "JSLAB_MLS_Filter.h"
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Analytical eigendecomposition of a 3x3 real symmetric matrix.
// Returns eigenvalues in ascending order and their corresponding eigenvectors
// (columns of evecs).  Uses Cardano's formula for the cubic characteristic
// polynomial, which is exact and branch-free for 3x3 symmetric matrices.
// ---------------------------------------------------------------------------
static void symmetricEigen3x3(const float C[3][3],
                               float evals[3], float evecs[3][3])
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
                    std::swap(evals[i], evals[j]);
                    std::swap(evecs[i][0], evecs[j][0]);
                    std::swap(evecs[i][1], evecs[j][1]);
                    std::swap(evecs[i][2], evecs[j][2]);
                }
        return;
    }

    const float trace = C[0][0] + C[1][1] + C[2][2];
    const float q = trace / 3.0f;

    float B[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            B[i][j] = C[i][j];
    B[0][0] -= q; B[1][1] -= q; B[2][2] -= q;

    float p2 = B[0][0] * B[0][0] + B[1][1] * B[1][1] + B[2][2] * B[2][2]
             + 2.0f * p1;
    float p = std::sqrt(p2 / 6.0f);
    float inv_p = (p > 1e-15f) ? (1.0f / p) : 0.0f;

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            B[i][j] *= inv_p;

    float detB = B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1])
               - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
               + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]);
    float r = detB * 0.5f;
    r = std::max(-1.0f, std::min(1.0f, r));

    float phi = std::acos(r) / 3.0f;
    const float PI = 3.14159265358979323846f;

    evals[0] = q + 2.0f * p * std::cos(phi + 2.0f * PI / 3.0f);
    evals[2] = q + 2.0f * p * std::cos(phi);
    evals[1] = trace - evals[0] - evals[2];

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

        float* best = c12;
        float  best_len = len12;
        if (len01 > best_len) { best = c01; best_len = len01; }
        if (len02 > best_len) { best = c02; best_len = len02; }

        float inv_len = (best_len > 1e-20f) ? (1.0f / std::sqrt(best_len)) : 0.0f;
        evecs[idx][0] = best[0] * inv_len;
        evecs[idx][1] = best[1] * inv_len;
        evecs[idx][2] = best[2] * inv_len;
    }
}

// ---------------------------------------------------------------------------
// Cholesky solver for small (nc x nc, nc <= 6) symmetric positive-definite
// systems.  Returns false if the matrix is not SPD (caller should fall back
// to a lower polynomial order).
// ---------------------------------------------------------------------------
static bool choleskySolve(const float A[][6], const float b[], float x[], int nc)
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
                L[i][j] = std::sqrt(s);
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
// Single-pass CPU MLS – called by the public multi-iteration wrapper.
// ---------------------------------------------------------------------------
static void applyMLSFilterToPointCloud_CPU_SinglePass(
    const cv::Mat& src_xyz, const cv::Mat& P,
    cv::Mat& dst_xyz, int window_size, float spatial_sigma,
    int polynomial_order, float radius2_cutoff)
{
    dst_xyz = src_xyz.clone();

    const int rows = src_xyz.rows;
    const int cols = src_xyz.cols;
    const int hw   = window_size;
    const float inv_2sigma2 = 1.0f / (2.0f * spatial_sigma * spatial_sigma);

    const int min_neighbors_by_order[3] = { 3, 5, 8 };
    const int order = std::max(0, std::min(2, polynomial_order));

#pragma omp parallel for collapse(2)
    for (int irow = 0; irow < rows; irow++)
    {
        for (int jcol = 0; jcol < cols; jcol++)
        {
            if (P.ptr<uchar>(irow)[jcol] == 0)
                continue;

            const cv::Vec3f& pi = src_xyz.ptr<cv::Vec3f>(irow)[jcol];

            // ---- 1. Weighted centroid ----
            float cx = 0, cy = 0, cz = 0;
            float w_sum = 0;
            int   n_valid = 0;

            int r0 = std::max(0, irow - hw);
            int r1 = std::min(rows, irow + hw + 1);
            int c0 = std::max(0, jcol - hw);
            int c1 = std::min(cols, jcol + hw + 1);

            for (int ir = r0; ir < r1; ir++)
            {
                const cv::Vec3f* xyz_row = src_xyz.ptr<cv::Vec3f>(ir);
                const uchar*     P_row   = P.ptr<uchar>(ir);
                for (int jc = c0; jc < c1; jc++)
                {
                    if (P_row[jc] == 0)
                        continue;

                    const cv::Vec3f& pk = xyz_row[jc];
                    float dx = pk[0] - pi[0];
                    float dy = pk[1] - pi[1];
                    float dz = pk[2] - pi[2];
                    float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 > radius2_cutoff)
                        continue;
                    float w = std::exp(-dist2 * inv_2sigma2);

                    cx += w * pk[0];
                    cy += w * pk[1];
                    cz += w * pk[2];
                    w_sum += w;
                    n_valid++;
                }
            }

            if (n_valid < min_neighbors_by_order[0] || w_sum < 1e-12f)
                continue;

            float inv_w = 1.0f / w_sum;
            cx *= inv_w;
            cy *= inv_w;
            cz *= inv_w;

            // ---- 2. Weighted covariance matrix (symmetric 3x3) ----
            float cov[3][3] = {};

            for (int ir = r0; ir < r1; ir++)
            {
                const cv::Vec3f* xyz_row = src_xyz.ptr<cv::Vec3f>(ir);
                const uchar*     P_row   = P.ptr<uchar>(ir);
                for (int jc = c0; jc < c1; jc++)
                {
                    if (P_row[jc] == 0)
                        continue;

                    const cv::Vec3f& pk = xyz_row[jc];
                    float dx = pk[0] - pi[0];
                    float dy = pk[1] - pi[1];
                    float dz = pk[2] - pi[2];
                    float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 > radius2_cutoff)
                        continue;
                    float w = std::exp(-dist2 * inv_2sigma2);

                    float ex = pk[0] - cx;
                    float ey = pk[1] - cy;
                    float ez = pk[2] - cz;

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
            symmetricEigen3x3(cov, evals, evecs);

            float nx  = evecs[0][0], ny  = evecs[0][1], nz  = evecs[0][2];
            float t1x = evecs[1][0], t1y = evecs[1][1], t1z = evecs[1][2];
            float t2x = evecs[2][0], t2y = evecs[2][1], t2z = evecs[2][2];

            // ---- 4. Polynomial surface fitting (with automatic order fallback) ----
            int cur_order = (n_valid >= min_neighbors_by_order[order]) ? order : 0;
            if (cur_order > 0 && n_valid < min_neighbors_by_order[cur_order])
                cur_order = (n_valid >= min_neighbors_by_order[1]) ? 1 : 0;

            cv::Vec3f& out = dst_xyz.ptr<cv::Vec3f>(irow)[jcol];

            if (cur_order == 0)
            {
                float dot = (pi[0] - cx) * nx + (pi[1] - cy) * ny + (pi[2] - cz) * nz;
                out[0] = pi[0] - dot * nx;
                out[1] = pi[1] - dot * ny;
                out[2] = pi[2] - dot * nz;
                continue;
            }

            // nc = number of polynomial coefficients
            const int nc = (cur_order == 1) ? 3 : 6;

            // Build weighted normal equations: (A^T W A) c = A^T W f
            float AtWA[6][6] = {};
            float AtWf[6]    = {};

            for (int ir = r0; ir < r1; ir++)
            {
                const cv::Vec3f* xyz_row = src_xyz.ptr<cv::Vec3f>(ir);
                const uchar*     P_row   = P.ptr<uchar>(ir);
                for (int jc = c0; jc < c1; jc++)
                {
                    if (P_row[jc] == 0)
                        continue;

                    const cv::Vec3f& pk = xyz_row[jc];
                    float dx = pk[0] - pi[0];
                    float dy = pk[1] - pi[1];
                    float dz = pk[2] - pi[2];
                    float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 > radius2_cutoff)
                        continue;
                    float w = std::exp(-dist2 * inv_2sigma2);

                    float ex = pk[0] - cx;
                    float ey = pk[1] - cy;
                    float ez = pk[2] - cz;
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

            // ---- 5. Solve – fall back to lower order on failure ----
            float coeffs[6] = {};
            bool solved = choleskySolve(AtWA, AtWf, coeffs, nc);

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
                solved = choleskySolve(AtWA1, AtWf1, coeffs, 3);
                if (solved)
                    cur_order = 1;
            }

            if (!solved)
            {
                float dot = (pi[0] - cx) * nx + (pi[1] - cy) * ny + (pi[2] - cz) * nz;
                out[0] = pi[0] - dot * nx;
                out[1] = pi[1] - dot * ny;
                out[2] = pi[2] - dot * nz;
                continue;
            }

            // ---- 6. Evaluate polynomial at query point ----
            float ex0 = pi[0] - cx, ey0 = pi[1] - cy, ez0 = pi[2] - cz;
            float u0 = ex0 * t1x + ey0 * t1y + ez0 * t1z;
            float v0 = ex0 * t2x + ey0 * t2y + ez0 * t2z;

            float f_fitted = coeffs[0] + coeffs[1] * u0 + coeffs[2] * v0;
            if (cur_order == 2)
                f_fitted += coeffs[3] * u0 * u0 + coeffs[4] * u0 * v0 + coeffs[5] * v0 * v0;

            // ---- 7. Transform back to world coordinates ----
            out[0] = cx + u0 * t1x + v0 * t2x + f_fitted * nx;
            out[1] = cy + u0 * t1y + v0 * t2y + f_fitted * ny;
            out[2] = cz + u0 * t1z + v0 * t2z + f_fitted * nz;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU MLS filter – polynomial surface projection on organized point cloud
// ---------------------------------------------------------------------------
void applyMLSFilterToPointCloud_CPU(
    const cv::Mat& src_xyz, const cv::Mat& P,
    cv::Mat& dst_xyz, int window_size, float spatial_sigma,
    int polynomial_order, float max_radius, int iterations)
{
    const float radius2_cutoff = (max_radius > 0.0f)
        ? max_radius * max_radius
        : 9.0f * spatial_sigma * spatial_sigma;
    const int n_iter = std::max(1, iterations);

    cv::Mat cur_input = src_xyz;
    for (int iter = 0; iter < n_iter; iter++)
    {
        applyMLSFilterToPointCloud_CPU_SinglePass(
            cur_input, P, dst_xyz, window_size, spatial_sigma,
            polynomial_order, radius2_cutoff);
        if (iter + 1 < n_iter)
            cur_input = dst_xyz.clone();
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatcher
// ---------------------------------------------------------------------------
void applyMLSFilterToPointCloud(
    const cv::Mat& src_xyz, const cv::Mat& P,
    cv::Mat& dst_xyz, int window_size, float spatial_sigma,
    bool use_gpu, int polynomial_order, float max_radius, int iterations)
{
#ifdef JSLAB_USE_CUDA
    if (use_gpu)
    {
        applyMLSFilterToPointCloud_GPU(src_xyz, P, dst_xyz,
                                       window_size, spatial_sigma,
                                       polynomial_order, max_radius,
                                       iterations);
        return;
    }
#else
    (void)use_gpu;
#endif
    applyMLSFilterToPointCloud_CPU(src_xyz, P, dst_xyz,
                                   window_size, spatial_sigma,
                                   polynomial_order, max_radius,
                                   iterations);
}
