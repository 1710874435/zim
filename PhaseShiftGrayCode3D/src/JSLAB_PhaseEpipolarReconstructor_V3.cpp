#include "JSLAB_PhaseEpipolarReconstructor_V3.h"
#include "JSLAB_MLS_Filter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
void computeFullLeftRectification(const cv::Mat& K1, const cv::Mat& D1,
	const cv::Size& image_size, const cv::Mat& R1,
	cv::Mat& P1, cv::Mat& P2, cv::Mat& Q, cv::Size& rectified_size)
{
	std::vector<cv::Point2f> corners = {
		{0.0f, 0.0f},
		{static_cast<float>(image_size.width - 1), 0.0f},
		{static_cast<float>(image_size.width - 1), static_cast<float>(image_size.height - 1)},
		{0.0f, static_cast<float>(image_size.height - 1)}
	};

	std::vector<cv::Point2f> rectified_corners;
	cv::undistortPoints(corners, rectified_corners, K1, D1, R1, P1);

	float min_x = rectified_corners[0].x;
	float max_x = rectified_corners[0].x;
	float min_y = rectified_corners[0].y;
	float max_y = rectified_corners[0].y;

	for (const cv::Point2f& point : rectified_corners)
	{
		min_x = std::min(min_x, point.x);
		max_x = std::max(max_x, point.x);
		min_y = std::min(min_y, point.y);
		max_y = std::max(max_y, point.y);
	}

	const double shift_x = -std::floor(min_x);
	const double shift_y = -std::floor(min_y);

	rectified_size = cv::Size(
		static_cast<int>(std::ceil(max_x - min_x + 1.0f)),
		static_cast<int>(std::ceil(max_y - min_y + 1.0f)));

	P1.at<double>(0, 2) += shift_x;
	P1.at<double>(1, 2) += shift_y;
	P2.at<double>(0, 2) += shift_x;
	P2.at<double>(1, 2) += shift_y;

	Q.at<double>(0, 3) -= shift_x;
	Q.at<double>(1, 3) -= shift_y;
}
}

void applyGaussianFilterToPointCloudV3(const cv::Mat& src_xyz, const cv::Mat& P,
	cv::Mat& dst_xyz, const int win_size)
{
	const float count_threshold = 0.8f;
	const int half_win_size = (win_size - 1) / 2;

	cv::Mat kerner1d = cv::getGaussianKernel(win_size, 0.7, CV_32FC1);
	cv::Mat kernel = kerner1d * kerner1d.t();
	dst_xyz = cv::Mat::zeros(src_xyz.size(), CV_32FC3);

#pragma omp parallel for collapse(2)
	for (int irow = half_win_size; irow < src_xyz.rows - half_win_size; irow++)
	{
		const cv::Vec3f* xyz_ptr = src_xyz.ptr<cv::Vec3f>(irow);
		cv::Vec3f* dst_xyz_ptr = dst_xyz.ptr<cv::Vec3f>(irow);
		const uchar* P_ptr = P.ptr<uchar>(irow);

		for (int jcol = half_win_size; jcol < src_xyz.cols - half_win_size; jcol++)
		{
			if (P_ptr[jcol] == 0)
				continue;

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
						continue;
					if (abs(xyz_ptr0[jcol + jk][0] - xyz_ptr[jcol][0]) < 3.0)
					{
						sum += xyz_ptr0[jcol + jk] * kernel_ptr[half_win_size + jk];
						count += kernel_ptr[half_win_size + jk];
					}
				}
			}

			if (count > count_threshold)
				dst_xyz_ptr[jcol] = sum / count;
		}
	}
}

void applyRadiusOutlierFilterToPointCloudV3(const cv::Mat& src_xyz, const cv::Mat& P,
	cv::Mat& dst_xyz, cv::Mat& dst_P,
	const float search_radius, const int min_neighbors,
	const int window_size = 15)
{
	dst_xyz = cv::Mat::zeros(src_xyz.size(), CV_32FC3);
	dst_P = cv::Mat::zeros(P.size(), CV_8UC1);

	const float radius_squared = search_radius * search_radius;
	const int half_win_size = window_size / 2;

#pragma omp parallel for collapse(2)
	for (int irow = 0; irow < src_xyz.rows; irow++)
	{
		for (int jcol = 0; jcol < src_xyz.cols; jcol++)
		{
			const uchar* P_ptr = P.ptr<uchar>(irow);

			if (P_ptr[jcol] == 0)
				continue;

			const cv::Vec3f* xyz_ptr = src_xyz.ptr<cv::Vec3f>(irow);
			const cv::Vec3f& current_point = xyz_ptr[jcol];

			int neighbor_count = 0;

			int row_start = std::max(0, irow - half_win_size);
			int row_end = std::min(src_xyz.rows, irow + half_win_size + 1);
			int col_start = std::max(0, jcol - half_win_size);
			int col_end = std::min(src_xyz.cols, jcol + half_win_size + 1);

			for (int i = row_start; i < row_end; i++)
			{
				const cv::Vec3f* xyz_ptr_i = src_xyz.ptr<cv::Vec3f>(i);
				const uchar* P_ptr_i = P.ptr<uchar>(i);

				for (int j = col_start; j < col_end; j++)
				{
					if (P_ptr_i[j] == 0)
						continue;
					if (i == irow && j == jcol)
						continue;

					const cv::Vec3f& neighbor_point = xyz_ptr_i[j];

					float dx = current_point[0] - neighbor_point[0];
					float dy = current_point[1] - neighbor_point[1];
					float dz = current_point[2] - neighbor_point[2];
					float dist_squared = dx * dx + dy * dy + dz * dz;

					if (dist_squared <= radius_squared)
					{
						neighbor_count++;
						if (neighbor_count >= min_neighbors)
							goto keep_point;
					}
				}
			}

			if (neighbor_count >= min_neighbors)
			{
keep_point:
				cv::Vec3f* dst_xyz_ptr = dst_xyz.ptr<cv::Vec3f>(irow);
				uchar* dst_P_ptr = dst_P.ptr<uchar>(irow);
				dst_xyz_ptr[jcol] = current_point;
				dst_P_ptr[jcol] = 1;
			}
		}
	}
}

JSLAB_PhaseEpipolarReconstructor_V3::JSLAB_PhaseEpipolarReconstructor_V3(
	const Config& config) :
	m_config(config),
	m_pha_f(0),
	m_is_init(false),
	m_min_disparity(0),
	m_max_disparity(0)
{
}

JSLAB_PhaseEpipolarReconstructor_V3::~JSLAB_PhaseEpipolarReconstructor_V3()
{
}

void JSLAB_PhaseEpipolarReconstructor_V3::estimateDisparityRange()
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

			double z1 = m_R1.at<double>(2, 0) * p1[0] +
				m_R1.at<double>(2, 1) * p1[1] +
				m_R1.at<double>(2, 2) * p1[2];
			double z2 = m_R1.at<double>(2, 0) * p2[0] +
				m_R1.at<double>(2, 1) * p2[1] +
				m_R1.at<double>(2, 2) * p2[2];

			if (z1 < z_min_rectification) z_min_rectification = z1;
			if (z1 > z_max_rectification) z_max_rectification = z1;
			if (z2 > z_max_rectification) z_max_rectification = z2;
			if (z2 < z_min_rectification) z_min_rectification = z2;
		}
	}

	m_min_disparity =
		(m_Q.at<double>(2, 3) /
			(z_max_rectification * m_Q.at<double>(3, 2))) -
		(m_Q.at<double>(3, 3) / m_Q.at<double>(3, 2));
	m_max_disparity =
		(m_Q.at<double>(2, 3) /
			(z_min_rectification * m_Q.at<double>(3, 2))) -
		(m_Q.at<double>(3, 3) / m_Q.at<double>(3, 2));
}

void JSLAB_PhaseEpipolarReconstructor_V3::setCalibrationParams(
	const JSLAB_StereoCalibParams_V1& params)
{
	m_params.camera_matrix[0] = params.camera_matrix[0].clone();
	m_params.camera_matrix[1] = params.camera_matrix[1].clone();
	m_params.dist_coeffs[0] = params.dist_coeffs[0].clone();
	m_params.dist_coeffs[1] = params.dist_coeffs[1].clone();
	m_params.R = params.R.clone();
	m_params.t = params.t.clone();
	m_params.image_size = params.image_size;

	cv::Rect rect1, rect2;
	cv::Mat R2, P1, P2;
	cv::stereoRectify(
		m_params.camera_matrix[0], m_params.dist_coeffs[0],
		m_params.camera_matrix[1], m_params.dist_coeffs[1],
		m_params.image_size, m_params.R, m_params.t,
		m_R1, R2, P1, P2, m_Q, 0, -1.0,
		m_params.image_size, &rect1, &rect2);

	cv::Size output_size = m_params.image_size;
	if (m_config.use_full_left_rectification)
	{
		computeFullLeftRectification(
			m_params.camera_matrix[0], m_params.dist_coeffs[0],
			m_params.image_size, m_R1, P1, P2, m_Q, output_size);
	}

	cv::initUndistortRectifyMap(
		m_params.camera_matrix[0], m_params.dist_coeffs[0],
		m_R1, P1, output_size, CV_32FC1,
		m_left_remap[0], m_left_remap[1]);
	cv::initUndistortRectifyMap(
		m_params.camera_matrix[1], m_params.dist_coeffs[1],
		R2, P2, output_size, CV_32FC1,
		m_right_remap[0], m_right_remap[1]);
}

void JSLAB_PhaseEpipolarReconstructor_V3::setData(
	const cv::Mat& left, const cv::Mat& right,
	const cv::Mat& left_k, const cv::Mat& right_k,
	const cv::Mat& left_mod, int pha_f)
{
	cv::remap(left, m_left,
		m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);
	cv::remap(right, m_right,
		m_right_remap[0], m_right_remap[1], cv::INTER_LINEAR);

	cv::remap(left_k, m_left_k,
		m_left_remap[0], m_left_remap[1], cv::INTER_NEAREST);
	cv::remap(right_k, m_right_k,
		m_right_remap[0], m_right_remap[1], cv::INTER_NEAREST);

	cv::remap(left_mod, m_left_mod,
		m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);

	m_pha_f = pha_f;

	const cv::Size output_size = m_left.size();
	if (!m_is_init || m_xyz.size() != output_size || m_P.size() != output_size)
	{
		m_xyz = cv::Mat(output_size, CV_32FC3, cv::Scalar::all(0.0));
		m_P = cv::Mat(output_size, CV_8UC1, cv::Scalar(0));
		m_is_init = true;
	}

	m_xyz = cv::Vec3f(0, 0, 0);
	m_P = 0;
}

void JSLAB_PhaseEpipolarReconstructor_V3::startCalculate(
	cv::Mat& xyz, cv::Mat& P)
{
	cv::Mat disp_map;
	stereoMatch(m_left, m_right, m_left_k, m_right_k, m_pha_f, disp_map);

	if (m_config.use_disp_median_filter)
	{
		disp_map.convertTo(disp_map, CV_32FC1);
		cv::medianBlur(disp_map, disp_map, 5);
		disp_map.convertTo(disp_map, CV_64FC1);
	}

	m_disp = disp_map.clone();

#pragma omp parallel for
	for (int irow = 0; irow < disp_map.rows; irow++)
	{
		const double* disp_ptr = disp_map.ptr<double>(irow);
		cv::Vec3f* xyz_ptr = m_xyz.ptr<cv::Vec3f>(irow);
		uchar* P_ptr = m_P.ptr<uchar>(irow);

		for (int jcol = 0; jcol < disp_map.cols; jcol++)
		{
			double d = disp_ptr[jcol];

			if (std::abs(d) < 1e-5)
				continue;

			double w = m_Q.at<double>(3, 2) * d +
				m_Q.at<double>(3, 3);
			double x = jcol + m_Q.at<double>(0, 3);
			double y = irow + m_Q.at<double>(1, 3);
			double z = m_Q.at<double>(2, 3);

			double inv_w = 1.0 / w;
			x *= inv_w;
			y *= inv_w;
			z *= inv_w;

			double rx = m_R1.at<double>(0, 0) * x +
				m_R1.at<double>(1, 0) * y +
				m_R1.at<double>(2, 0) * z;
			double ry = m_R1.at<double>(0, 1) * x +
				m_R1.at<double>(1, 1) * y +
				m_R1.at<double>(2, 1) * z;
			double rz = m_R1.at<double>(0, 2) * x +
				m_R1.at<double>(1, 2) * y +
				m_R1.at<double>(2, 2) * z;

			if (m_config.use_rectified_coords)
			{
				if (z > m_config.min_depth && z < m_config.max_depth)
				{
					xyz_ptr[jcol] = cv::Vec3f(
						static_cast<float>(x),
						static_cast<float>(y),
						static_cast<float>(z));
					P_ptr[jcol] = 1;
				}
			}
			else
			{
				if (rz > m_config.min_depth && rz < m_config.max_depth)
				{
					xyz_ptr[jcol] = cv::Vec3f(
						static_cast<float>(rx),
						static_cast<float>(ry),
						static_cast<float>(rz));
					P_ptr[jcol] = 1;
				}
			}
		}
	}

	applyGaussianFilterToPointCloudV3(m_xyz, m_P, xyz, 3);

	cv::Mat xyz_filtered, P_filtered;
	applyRadiusOutlierFilterToPointCloudV3(
		xyz, m_P, xyz_filtered, P_filtered,
		m_config.radius_search,
		m_config.radius_min_neighbors,
		m_config.radius_window_size);
	xyz = xyz_filtered;
	P = P_filtered;

	if (m_config.use_mls)
	{
		cv::Mat xyz_mls;
		applyMLSFilterToPointCloud(
			xyz, P, xyz_mls,
			m_config.mls_window_size,
			m_config.mls_spatial_sigma,
			m_config.use_gpu_mls,
			m_config.mls_polynomial_order,
			m_config.mls_max_radius,
			m_config.mls_iterations);
		xyz = xyz_mls;
	}
}

void JSLAB_PhaseEpipolarReconstructor_V3::getDisparityMap(
	cv::Mat& disp) const
{
	disp = m_disp.clone();
}

void JSLAB_PhaseEpipolarReconstructor_V3::stereoMatch(
	const cv::Mat& l_phase, const cv::Mat& r_phase,
	const cv::Mat& l_k, const cv::Mat& r_k,
	int pha_f, cv::Mat& disp)
{
	disp = cv::Mat::zeros(l_phase.size(), CV_64FC1);

	const int max_rank_limit = m_pha_f;
	const int knum = m_pha_f;

#pragma omp parallel for
	for (int i = 0; i < l_phase.rows; i++)
	{
		const double* l_phase_ptr = l_phase.ptr<double>(i);
		const double* r_phase_ptr = r_phase.ptr<double>(i);
		const int* l_k_ptr = l_k.ptr<int>(i);
		const int* r_k_ptr = r_k.ptr<int>(i);
		double* disp_ptr = disp.ptr<double>(i);

		std::vector<std::vector<int>> row_rank_table(knum + 1);
		for (int j = 0; j < r_phase.cols; j++)
		{
			int k_val = r_k_ptr[j];
			if (k_val >= 0 && k_val <= knum)
				row_rank_table[k_val].push_back(j);
		}

		for (int j = 0; j < l_phase.cols; j++)
		{
			int k = l_k_ptr[j];

			if (k > max_rank_limit || k <= 0)
				continue;
			if (k > knum)
				continue;

			if (row_rank_table[k].empty())
				continue;

			double min_cost = 1e9;
			int best_jR = -1;
			double l_val = l_phase_ptr[j];
			const std::vector<int>& candidates = row_rank_table[k];

			for (int cand_idx : candidates)
			{
				double diff = std::abs(r_phase_ptr[cand_idx] - l_val);
				if (diff < min_cost)
				{
					min_cost = diff;
					best_jR = cand_idx;
				}
			}

			if (min_cost >= m_config.grad_threshold)
				continue;

			if (best_jR < 2 || best_jR >= r_phase.cols - 1)
				continue;

			double cost0 = r_phase_ptr[best_jR - 1] - l_val;
			double cost1 = r_phase_ptr[best_jR] - l_val;
			double cost2 = r_phase_ptr[best_jR + 1] - l_val;

			double numeratorL = cost1 * (cost0 - cost2);
			double denominatorL =
				2.0 * (cost0 - cost1) * (cost1 - cost2);

			double delta_L = 0.0;
			if (std::abs(denominatorL) > 1e-9)
				delta_L = numeratorL / denominatorL;

			disp_ptr[j] =
				static_cast<double>(j) -
				static_cast<double>(best_jR) -
				delta_L;
		}
	}
}
