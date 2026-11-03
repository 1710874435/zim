#pragma once

#include <opencv.hpp>
#include "JSLAB_StereoCalibParams_V1.h"
#include "dllexport.h"

/**
  * @brief GPU (CUDA) port of the V2 phase epipolar reconstructor.
  *
  * Same algorithm and public interface as JSLAB_PhaseEpipolarReconstructor_V2,
  * but the per-frame heavy pipeline (stereo matching, reproject-to-3D, point
  * cloud Gaussian filter, radius outlier filter) runs on the device through
  * custom CUDA kernels.  Calibration / rectification setup (stereoRectify,
  * initUndistortRectifyMap, remap) stays on the CPU.
  *
  * Precision: single precision (float) everywhere except the Q-matrix
  * perspective divide and the R1 rotation inside the reproject kernel, which
  * use double to avoid overflow / precision loss.
  */
class DLL_EXPORT JSLAB_PhaseEpipolarReconstructor_V2_GPU
{
public:

	/**
	  * @brief Configuration parameters (identical to the CPU V2 reconstructor)
	  */
	struct Config
	{
		// Depth range clamp (camera-coordinate Z, same unit as calibration)
		double min_depth = 0;
		double max_depth = 1000;

		// Adjacent-pixel phase difference limits used to filter unreliable pixels
		double min_phase_diff = 0.03;
		double max_phase_diff = 0.4;

		// Maximum allowed phase residual between matched left and right pixels
		double grad_threshold = 1;

		// When true:  output x/y/z (rectified coordinate system) and filter on z.
		// When false: output rx/ry/rz (original camera coordinate system, R1^T applied)
		//             and filter on rz.
		bool use_rectified_coords = true;

		// When true: apply a 5x5 median blur to the disparity map before reprojection.
		bool use_disp_median_filter = true;

		// Radius outlier filter parameters
		float radius_search        = 0.05f;  // 3-D search radius
		int   radius_min_neighbors = 5;      // min neighbours to keep a point
		int   radius_window_size   = 5;      // pixel window half-size for neighbour search

		// MLS smoothing (applied after radius outlier filter)
		bool  use_mls              = true;
		int   mls_window_size      = 25;
		float mls_spatial_sigma    = 0.5f;
		bool  use_gpu_mls          = true;
		int   mls_polynomial_order = 1;
		float mls_max_radius       = 0.0f;
		int   mls_iterations       = 2;
	};

	JSLAB_PhaseEpipolarReconstructor_V2_GPU(const Config& config);

	~JSLAB_PhaseEpipolarReconstructor_V2_GPU();

	/**
	  * @brief Set calibration parameters (triggers rectification map computation)
	  */
	void setCalibrationParams(const JSLAB_StereoCalibParams_V1& params);

	/**
	  * @brief Set phase maps, order maps and modulation map
	  */
	void setData(const cv::Mat& left, const cv::Mat& right,
	             const cv::Mat& left_k, const cv::Mat& right_k,
	             const cv::Mat& left_mod, int pha_f);

	/**
	  * @brief Run 3-D reconstruction on the GPU
	  * @param [out] xyz Output 3-D point cloud (CV_32FC3)
	  * @param [out] P   Valid pixel mask (CV_8UC1, non-zero = valid)
	  */
	void startCalculate(cv::Mat& xyz, cv::Mat& P);

	/**
	  * @brief [Diagnostic] Get the disparity map computed during the last
	  *        startCalculate (after stereo match and optional median blur).
	  * @param [out] disp Disparity map (CV_32FC1)
	  */
	void getDisparityMap(cv::Mat& disp) const;

private:

	/**
	  * @brief Estimate the valid disparity range from the configured depth bounds
	  */
	void estimateDisparityRange();

private:

	const Config m_config;
	JSLAB_StereoCalibParams_V1 m_params;
	cv::Mat m_left_remap[2];
	cv::Mat m_right_remap[2];
	cv::Mat m_R1;
	cv::Mat m_Q;
	cv::Mat m_left;
	cv::Mat m_right;
	cv::Mat m_left_k;      // Rectified left fringe order map
	cv::Mat m_right_k;     // Rectified right fringe order map
	cv::Mat m_left_mod;    // Rectified left modulation map
	int m_pha_f;           // Fringe frequency parameter
	bool m_is_init;
	double m_min_disparity;
	double m_max_disparity;
	cv::Mat m_disp;        // [Diagnostic] disparity map from the last run
};

// ---------------------------------------------------------------------------
// CUDA host launcher (implemented in JSLAB_PhaseEpipolarReconstructor_V2_GPU.cu).
//
// Uploads the rectified phase / order maps once, runs the whole pipeline
// (stereo match -> optional 5x5 median -> reproject-to-3D -> point-cloud
// Gaussian filter -> radius outlier filter) device-to-device, and downloads
// the filtered point cloud + mask.  Q and R1 are passed row-major as double.
//   xyz      : CV_32FC3
//   P        : CV_8UC1
//   disp_map : CV_32FC1 [diagnostic] disparity after stereo match + median
// ---------------------------------------------------------------------------
void JSLAB_ReconstructPipeline_GPU(
    const cv::Mat& l_phase, const cv::Mat& r_phase,
    const cv::Mat& l_k, const cv::Mat& r_k,
    int pha_f, double min_disp, double max_disp,
    const double Q[16], const double R1[9],
    const JSLAB_PhaseEpipolarReconstructor_V2_GPU::Config& cfg,
    cv::Mat& xyz, cv::Mat& P, cv::Mat& disp_map);
