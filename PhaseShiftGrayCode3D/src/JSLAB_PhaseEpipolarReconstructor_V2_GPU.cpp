#include "JSLAB_PhaseEpipolarReconstructor_V2_GPU.h"
#include "JSLAB_MLS_Filter.h"
#include <vector>

JSLAB_PhaseEpipolarReconstructor_V2_GPU::JSLAB_PhaseEpipolarReconstructor_V2_GPU(const Config& config) :
	m_config(config),
	m_pha_f(0),
	m_is_init(false),
	m_min_disparity(0),
	m_max_disparity(0)
{

}

JSLAB_PhaseEpipolarReconstructor_V2_GPU::~JSLAB_PhaseEpipolarReconstructor_V2_GPU()
{

}

void JSLAB_PhaseEpipolarReconstructor_V2_GPU::estimateDisparityRange()
{
	double z_min_rectification = 1e3;
	double z_max_rectification = -1e3;
	const double fx = m_params.camera_matrix[0].at<double>(0, 0);
	const double fy = m_params.camera_matrix[0].at<double>(1, 1);
	const double cx = m_params.camera_matrix[0].at<double>(0, 2);
	const double cy = m_params.camera_matrix[0].at<double>(1, 2);

	const double z_min = m_config.min_depth;
	const double z_max = m_config.max_depth;

	for (int irow = 0; irow < m_params.image_size.height; irow++)
	{
		for (int jcol = 0; jcol < m_params.image_size.width; jcol++)
		{
			double norm_x = (jcol - cx) / fx;
			double norm_y = (irow - cy) / fy;

			cv::Vec3d p1(z_min * norm_x, z_min * norm_y, z_min);
			cv::Vec3d p2(z_max * norm_x, z_max * norm_y, z_max);

			double z1 = m_R1.at<double>(2, 0) * p1[0] + m_R1.at<double>(2, 1) * p1[1] + m_R1.at<double>(2, 2) * p1[2];
			double z2 = m_R1.at<double>(2, 0) * p2[0] + m_R1.at<double>(2, 1) * p2[1] + m_R1.at<double>(2, 2) * p2[2];

			if (z1 < z_min_rectification) z_min_rectification = z1;
			if (z1 > z_max_rectification) z_max_rectification = z1;
			if (z2 > z_max_rectification) z_max_rectification = z2;
			if (z2 < z_min_rectification) z_min_rectification = z2;
		}
	}

	m_min_disparity = (m_Q.at<double>(2, 3) / (z_max_rectification * m_Q.at<double>(3, 2))) - (m_Q.at<double>(3, 3) / m_Q.at<double>(3, 2));
	m_max_disparity = (m_Q.at<double>(2, 3) / (z_min_rectification * m_Q.at<double>(3, 2))) - (m_Q.at<double>(3, 3) / m_Q.at<double>(3, 2));
}

void JSLAB_PhaseEpipolarReconstructor_V2_GPU::setCalibrationParams(const JSLAB_StereoCalibParams_V1& params)
{
	m_params.camera_matrix[0] = params.camera_matrix[0].clone();
	m_params.camera_matrix[1] = params.camera_matrix[1].clone();
	m_params.dist_coeffs[0]   = params.dist_coeffs[0].clone();
	m_params.dist_coeffs[1]   = params.dist_coeffs[1].clone();
	m_params.R                = params.R.clone();
	m_params.t                = params.t.clone();
	m_params.image_size       = params.image_size;

	cv::Rect rect1, rect2;
	cv::Mat R2, P1, P2;
	cv::stereoRectify(m_params.camera_matrix[0], m_params.dist_coeffs[0],
	                  m_params.camera_matrix[1], m_params.dist_coeffs[1],
	                  m_params.image_size, m_params.R, m_params.t,
	                  m_R1, R2, P1, P2, m_Q, 0, -1.0, m_params.image_size, &rect1, &rect2);

	cv::initUndistortRectifyMap(m_params.camera_matrix[0], m_params.dist_coeffs[0],
	                            m_R1, P1, m_params.image_size, CV_32FC1,
	                            m_left_remap[0], m_left_remap[1]);
	cv::initUndistortRectifyMap(m_params.camera_matrix[1], m_params.dist_coeffs[1],
	                            R2, P2, m_params.image_size, CV_32FC1,
	                            m_right_remap[0], m_right_remap[1]);

	// The GPU stereo match restricts its search to the disparity window, so we
	// must derive it from the configured depth range up front.
	estimateDisparityRange();
}

void JSLAB_PhaseEpipolarReconstructor_V2_GPU::setData(const cv::Mat& left, const cv::Mat& right,
                                                       const cv::Mat& left_k, const cv::Mat& right_k,
                                                       const cv::Mat& left_mod, int pha_f)
{
	cv::remap(left,  m_left,  m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);
	cv::remap(right, m_right, m_right_remap[0], m_right_remap[1], cv::INTER_LINEAR);

	cv::remap(left_k,  m_left_k,  m_left_remap[0], m_left_remap[1], cv::INTER_NEAREST);
	cv::remap(right_k, m_right_k, m_right_remap[0], m_right_remap[1], cv::INTER_NEAREST);

	cv::remap(left_mod, m_left_mod, m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);

	m_pha_f   = pha_f;
	m_is_init = true;
}

void JSLAB_PhaseEpipolarReconstructor_V2_GPU::startCalculate(cv::Mat& xyz, cv::Mat& P)
{
	// Flatten Q (4x4) and R1 (3x3) to row-major double arrays for the kernels.
	double Qarr[16];
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			Qarr[i * 4 + j] = m_Q.at<double>(i, j);

	double R1arr[9];
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			R1arr[i * 3 + j] = m_R1.at<double>(i, j);

	JSLAB_ReconstructPipeline_GPU(
		m_left, m_right, m_left_k, m_right_k,
		m_pha_f, m_min_disparity, m_max_disparity,
		Qarr, R1arr, m_config, xyz, P, m_disp);

	if (m_config.use_mls)
	{
		cv::Mat xyz_mls;
		applyMLSFilterToPointCloud(xyz, P, xyz_mls,
			m_config.mls_window_size, m_config.mls_spatial_sigma,
			m_config.use_gpu_mls, m_config.mls_polynomial_order,
			m_config.mls_max_radius, m_config.mls_iterations);
		xyz = xyz_mls;
	}
}

void JSLAB_PhaseEpipolarReconstructor_V2_GPU::getDisparityMap(cv::Mat& disp) const
{
	disp = m_disp.clone();
}
