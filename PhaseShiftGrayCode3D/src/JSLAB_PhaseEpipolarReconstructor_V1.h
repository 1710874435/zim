#pragma once

#include <opencv.hpp>
#include "JSLAB_StereoCalibParams_V1.h"
#include "dllexport.h"

class DLL_EXPORT JSLAB_PhaseEpipolarReconstructor_V1
{
public:

	/**
	  * @brief Configuration parameters
	  */
	struct Config
	{
		// Depth range clamp (camera-coordinate Z, same unit as calibration)
		double min_depth = 350.0;
		double max_depth = 700.0;

		// Adjacent-pixel phase difference limits used to filter unreliable pixels
		double min_phase_diff = 0.03;
		double max_phase_diff = 0.4;
	};

	JSLAB_PhaseEpipolarReconstructor_V1(const Config& config);

	~JSLAB_PhaseEpipolarReconstructor_V1();

	/**
	  * @brief Set calibration parameters (triggers rectification map computation)
	  * @param params Stereo calibration parameters
	  */
	void setCalibrationParams(const JSLAB_StereoCalibParams_V1& params);

	/**
	  * @brief Set input absolute phase maps
	  * @param left  Left camera absolute phase map (CV_64FC1)
	  * @param right Right camera absolute phase map (CV_64FC1)
	  */
	void setPhase(const cv::Mat& left, const cv::Mat& right);

	/**
	  * @brief Run 3-D reconstruction
	  * @param [out] xyz Output 3-D point cloud (CV_32FC3)
	  * @param [out] P   Valid pixel mask (CV_8UC1, non-zero = valid)
	  */
	void startCalculate(cv::Mat& xyz, cv::Mat& P);

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
	cv::Mat m_xyz;
	cv::Mat m_P;
	bool m_is_init;
	double m_min_disparity;
	double m_max_disparity;
};
