#pragma once

#include <opencv.hpp>
#include "JSLAB_StereoCalibParams_V1.h"
#include "dllexport.h"

/**
  * @brief V3 phase epipolar reconstructor.
  *
  * The reconstruction pipeline is identical to V2. V3 only adds an optional
  * full-left rectification canvas that preserves the complete left image.
  */
class DLL_EXPORT JSLAB_PhaseEpipolarReconstructor_V3
{
public:
	struct Config
	{
		double min_depth = 0;
		double max_depth = 1000;

		double min_phase_diff = 0.03;
		double max_phase_diff = 0.4;

		double grad_threshold = 1;

		bool use_rectified_coords = true;
		bool use_disp_median_filter = true;

		// false: V2-compatible output size (same as the source image).
		// true: preserve the complete rectified left image.
		bool use_full_left_rectification = true;

		float radius_search = 0.05f;
		int radius_min_neighbors = 5;
		int radius_window_size = 5;

		bool use_mls = true;
		int mls_window_size = 30;
		float mls_spatial_sigma = 0.6f;
		bool use_gpu_mls = true;
		int mls_polynomial_order = 1;
		float mls_max_radius = 0.0f;
		int mls_iterations = 3;
	};

	explicit JSLAB_PhaseEpipolarReconstructor_V3(const Config& config);
	~JSLAB_PhaseEpipolarReconstructor_V3();

	void setCalibrationParams(const JSLAB_StereoCalibParams_V1& params);

	void setData(const cv::Mat& left, const cv::Mat& right,
		const cv::Mat& left_k, const cv::Mat& right_k,
		const cv::Mat& left_mod, int pha_f);

	void startCalculate(cv::Mat& xyz, cv::Mat& P);
	void getDisparityMap(cv::Mat& disp) const;

private:
	void estimateDisparityRange();

	void stereoMatch(const cv::Mat& l_phase, const cv::Mat& r_phase,
		const cv::Mat& l_k, const cv::Mat& r_k,
		int pha_f, cv::Mat& disp);

private:
	const Config m_config;
	JSLAB_StereoCalibParams_V1 m_params;
	cv::Mat m_left_remap[2];
	cv::Mat m_right_remap[2];
	cv::Mat m_R1;
	cv::Mat m_Q;
	cv::Mat m_left;
	cv::Mat m_right;
	cv::Mat m_left_k;
	cv::Mat m_right_k;
	cv::Mat m_left_mod;
	int m_pha_f;
	cv::Mat m_xyz;
	cv::Mat m_P;
	cv::Mat m_disp;
	bool m_is_init;
	double m_min_disparity;
	double m_max_disparity;
};
