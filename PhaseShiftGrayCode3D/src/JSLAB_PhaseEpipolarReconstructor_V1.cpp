#include "JSLAB_PhaseEpipolarReconstructor_V1.h"

void applyGaussianFilterToPointCloud(const cv::Mat& src_xyz, const cv::Mat& P, cv::Mat& dst_xyz, const int win_size)
{
	const float count_threshold = 0.8f;
	const int half_win_size = (win_size - 1) / 2;

	// Build 2-D Gaussian kernel
	cv::Mat kerner1d = cv::getGaussianKernel(win_size, 0.7, CV_32FC1);
	cv::Mat kernel = kerner1d * kerner1d.t();
	dst_xyz = cv::Mat::zeros(src_xyz.size(), CV_32FC3);

#pragma omp parallel for
	for (int irow = half_win_size; irow < src_xyz.rows - half_win_size; irow++)
	{
		const cv::Vec3f* xyz_ptr = src_xyz.ptr<cv::Vec3f>(irow);
		cv::Vec3f* dst_xyz_ptr = dst_xyz.ptr<cv::Vec3f>(irow);
		const uchar* P_ptr = P.ptr<uchar>(irow);

#pragma omp parallel for
		for (int jcol = half_win_size; jcol < src_xyz.cols - half_win_size; jcol++)
		{
			if (P_ptr[jcol] == 0)
			{
				continue;
			}

			cv::Vec3f sum(0, 0, 0);
			float count = 0;
			for (int ik = -half_win_size; ik <= half_win_size; ik++)
			{
				const cv::Vec3f* xyz_ptr0 = src_xyz.ptr<cv::Vec3f>(irow + ik);
				const uchar* P_ptr0 = P.ptr<uchar>(irow + ik);
				const float* kernel_ptr = kernel.ptr<float>(ik + half_win_size);
				for (int jk = -half_win_size; jk <= half_win_size; jk++)
				{
					if (P_ptr0[jcol + jk] == 0)
					{
						continue;
					}
					if (abs(xyz_ptr0[jcol + jk][0] - xyz_ptr[jcol][0]) < 3.0)
					{
						sum += xyz_ptr0[jcol + jk] * kernel_ptr[half_win_size + jk];
						count += kernel_ptr[half_win_size + jk];
					}
				}
			}

			if (count > count_threshold)
			{
				dst_xyz_ptr[jcol] = sum / count;
			}
		}
	}
}



void JSLAB_PhaseEpipolarReconstructor_V1::estimateDisparityRange()
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
			double norm_x, norm_y;

			norm_x = (jcol - cx) / fx;
			norm_y = (irow - cy) / fy;

			cv::Vec3d p1(z_min * norm_x, z_min * norm_y, z_min);
			cv::Vec3d p2(z_max * norm_x, z_max * norm_y, z_max);

			double z1 = m_R1.at<double>(2, 0) * p1[0] + m_R1.at<double>(2, 1) * p1[1] + m_R1.at<double>(2, 2) * p1[2];
			double z2 = m_R1.at<double>(2, 0) * p2[0] + m_R1.at<double>(2, 1) * p2[1] + m_R1.at<double>(2, 2) * p2[2];

			if (z1 < z_min_rectification)
			{
				z_min_rectification = z1;
			}

			if (z1 > z_max_rectification)
			{
				z_max_rectification = z1;
			}

			if (z2 > z_max_rectification)
			{
				z_max_rectification = z2;
			}

			if (z2 < z_min_rectification)
			{
				z_min_rectification = z2;
			}
		}
	}

	m_min_disparity = (m_Q.at<double>(2, 3) / (z_max_rectification * m_Q.at<double>(3, 2))) - (m_Q.at<double>(3, 3) / m_Q.at<double>(3, 2));
	m_max_disparity = (m_Q.at<double>(2, 3) / (z_min_rectification * m_Q.at<double>(3, 2))) - (m_Q.at<double>(3, 3) / m_Q.at<double>(3, 2));

	std::cout << "min_disparity: " << m_min_disparity << std::endl;
	std::cout << "max_disparity: " << m_max_disparity << std::endl;
}


JSLAB_PhaseEpipolarReconstructor_V1::JSLAB_PhaseEpipolarReconstructor_V1(const Config& config) :
	m_config(config),
	m_is_init(false),
	m_min_disparity(0),
	m_max_disparity(0)
{

}

JSLAB_PhaseEpipolarReconstructor_V1::~JSLAB_PhaseEpipolarReconstructor_V1()
{

}

void JSLAB_PhaseEpipolarReconstructor_V1::setCalibrationParams(const JSLAB_StereoCalibParams_V1& params)
{
	m_params.camera_matrix[0] = params.camera_matrix[0].clone();
	m_params.camera_matrix[1] = params.camera_matrix[1].clone();
	m_params.dist_coeffs[0]   = params.dist_coeffs[0].clone();
	m_params.dist_coeffs[1]   = params.dist_coeffs[1].clone();
	m_params.R                = params.R.clone();
	m_params.t                = params.t.clone();
	m_params.image_size       = params.image_size;

	// Compute stereo rectification transforms
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

	// Estimate valid disparity range
	estimateDisparityRange();
}

void JSLAB_PhaseEpipolarReconstructor_V1::setPhase(const cv::Mat& left, const cv::Mat& right)
{
	// Apply epipolar rectification to both phase maps
	cv::remap(left,  m_left,  m_left_remap[0],  m_left_remap[1],  cv::INTER_LINEAR);
	cv::remap(right, m_right, m_right_remap[0], m_right_remap[1], cv::INTER_LINEAR);

	if (!m_is_init)
	{
		m_xyz    = cv::Mat(left.size(), CV_32FC3, cv::Scalar::all(0.0));
		m_P      = cv::Mat(left.size(), CV_8UC1,  cv::Scalar(0));
		m_is_init = true;
	}

	m_xyz = cv::Vec3f(0, 0, 0);
	m_P   = 0;
}

void JSLAB_PhaseEpipolarReconstructor_V1::startCalculate(cv::Mat& xyz, cv::Mat& P)
{
#pragma omp parallel for
	for (int irow = 0; irow < m_left.rows - 1; irow++)
	{
		const double* lphase_ptr  = m_left.ptr<double>(irow);
		const double* rphase_ptr  = m_right.ptr<double>(irow);
		const double* lphase_ptr1 = m_left.ptr<double>(irow + 1);
		const double* rphase_ptr1 = m_right.ptr<double>(irow + 1);

		cv::Vec3f* xyz_ptr = m_xyz.ptr<cv::Vec3f>(irow);
		uchar*     P_ptr   = m_P.ptr<uchar>(irow);

#pragma omp parallel for
		for (int jcol = 0; jcol < m_left.cols; jcol++)
		{
			double ldiff = lphase_ptr[jcol + 1] - lphase_ptr[jcol];
			if (lphase_ptr[jcol] < 1e-3 || lphase_ptr[jcol + 1] < 1e-3
				|| ldiff > m_config.max_phase_diff || ldiff < m_config.min_phase_diff
				|| abs(lphase_ptr1[jcol] - lphase_ptr[jcol]) > m_config.max_phase_diff)
			{
				continue;
			}

			int start = jcol - m_max_disparity;
			int end   = jcol - m_min_disparity;

			if (start > m_left.cols - 2 || end < 1)
				continue;
			if (start < 0)
				start = 0;
			if (end > m_left.cols - 1)
				end = m_left.cols - 1;

			double phase = lphase_ptr[jcol];

			for (int k = start; k < end; k++)
			{
				double rdiff = rphase_ptr[k + 1] - rphase_ptr[k];
				if (rphase_ptr[k] < 1e-3 || rphase_ptr[k + 1] < 1e-3
					|| rdiff > m_config.max_phase_diff || rdiff < m_config.min_phase_diff
					|| abs(rphase_ptr1[k] - rphase_ptr[k]) > m_config.max_phase_diff)
				{
					continue;
				}

				if ((rphase_ptr[k + 1] >= phase && rphase_ptr[k] <= phase))
				{
					double rcol = k + (rphase_ptr[k] - phase) / (rphase_ptr[k] - rphase_ptr[k + 1]);
					double d    = (double)jcol - rcol;

					double w = m_Q.at<double>(3, 2) * d + m_Q.at<double>(3, 3);
					double x = jcol + m_Q.at<double>(0, 3);
					double y = irow + m_Q.at<double>(1, 3);
					double z = m_Q.at<double>(2, 3);

					x /= w;
					y /= w;
					z /= w;

					// Transform to camera coordinate system
					double zz = m_R1.at<double>(0, 2) * x + m_R1.at<double>(1, 2) * y + m_R1.at<double>(2, 2) * z;

					if (zz > m_config.min_depth && zz < m_config.max_depth)
					{
						xyz_ptr[jcol] = cv::Vec3f(x, y, z);
						P_ptr[jcol]   = 1;
					}

					break;
				}
			}
		}
	}

	// Morphological erosion to remove isolated noise pixels
	cv::erode(m_P, m_P, cv::Mat(), cv::Point(-1, -1), 2);
	// Morphological dilation to restore valid regions
	cv::dilate(m_P, m_P, cv::Mat(), cv::Point(-1, -1), 2);

	// Gaussian smoothing on the point cloud
	applyGaussianFilterToPointCloud(m_xyz, m_P, xyz, 3);

	P = m_P.clone();
}
