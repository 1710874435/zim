#pragma once

#include "JSLAB_Config.h"
#include <opencv.hpp>
#include "dllexport.h"

/**
 * @brief CPU MLS smoothing for organized point clouds (OpenMP parallelized).
 *
 * For each valid pixel, builds a local coordinate frame via weighted PCA,
 * then fits a polynomial surface in the tangent plane and projects the point
 * onto that surface.
 *
 * Weight convention: w = exp(-d^2 / (2*sigma^2)), which is a standard
 * statistical Gaussian.  PCL uses exp(-d^2 / h^2) where h = search_radius,
 * so an equivalent PCL radius is sigma * sqrt(2).
 *
 * @param src_xyz          Input point cloud (CV_32FC3, image-aligned)
 * @param P                Validity mask     (CV_8UC1, non-zero = valid)
 * @param dst_xyz          Output smoothed point cloud (same size/type as src_xyz)
 * @param window_size      Half-size of the pixel search window (full window = 2*window_size+1)
 * @param spatial_sigma    Gaussian weight sigma in 3-D Euclidean space
 * @param polynomial_order 0 = plane projection, 1 = bilinear, 2 = biquadratic (default 0)
 * @param max_radius       Hard 3-D distance cutoff for neighbors.
 *                         0 = auto (3 * spatial_sigma).  Prevents cross-surface
 *                         contamination at depth discontinuities.
 * @param iterations       Number of MLS passes (>=1).  Each pass feeds the
 *                         smoothed output back as input.  2-3 passes match
 *                         CloudCompare / PCL quality.
 */
DLL_EXPORT void applyMLSFilterToPointCloud_CPU(
    const cv::Mat& src_xyz, const cv::Mat& P,
    cv::Mat& dst_xyz, int window_size, float spatial_sigma,
    int polynomial_order = 0, float max_radius = 0.0f,
    int iterations = 1);

#ifdef JSLAB_USE_CUDA
/**
 * @brief GPU MLS smoothing for organized point clouds (CUDA).
 *        Same semantics as the CPU version.
 */
DLL_EXPORT void applyMLSFilterToPointCloud_GPU(
    const cv::Mat& src_xyz, const cv::Mat& P,
    cv::Mat& dst_xyz, int window_size, float spatial_sigma,
    int polynomial_order = 0, float max_radius = 0.0f,
    int iterations = 1);
#endif

/**
 * @brief Runtime dispatcher – calls GPU version when use_gpu is true and CUDA
 *        support was compiled in; falls back to CPU otherwise.
 */
DLL_EXPORT void applyMLSFilterToPointCloud(
    const cv::Mat& src_xyz, const cv::Mat& P,
    cv::Mat& dst_xyz, int window_size, float spatial_sigma,
    bool use_gpu, int polynomial_order = 0, float max_radius = 0.0f,
    int iterations = 1);
