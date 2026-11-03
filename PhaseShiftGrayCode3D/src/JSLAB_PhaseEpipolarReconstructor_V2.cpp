#include "JSLAB_PhaseEpipolarReconstructor_V2.h"
#include "JSLAB_MLS_Filter.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
namespace {
constexpr float kInvDepthCoeffMarker = -1.0f;
constexpr float kFittedRayCoeffMarker = -2.0f;
constexpr float kZLinearXYCoeffMarker = -3.0f;
constexpr float kRationalDepthCoeffMarker = -4.0f;

bool evaluateV2CalibPoint(const cv::Vec4f& cx, const cv::Vec4f& cy, const cv::Vec4f& cz,
	                      double d, double& x_rect, double& y_rect, double& z_rect)
{
	if (std::abs(cz[0]) <= 1e-9 && std::abs(cz[1]) <= 1e-9 &&
		std::abs(cz[2]) <= 1e-9 && std::abs(cz[3]) <= 1e-9) {
		return false;
	}

	if (std::abs(cx[3] - kInvDepthCoeffMarker) < 1e-6f) {
		double d2 = d * d;
		double inv_z = cz[0] + cz[1] * d + cz[2] * d2 + cz[3] * d2 * d;
		if (!std::isfinite(inv_z) || std::abs(inv_z) < 1e-12) return false;
		z_rect = 1.0 / inv_z;
		x_rect = cx[0] * z_rect;
		y_rect = cy[0] * z_rect;
		return std::isfinite(x_rect) && std::isfinite(y_rect) && std::isfinite(z_rect);
	}

	if (std::abs(cx[3] - kFittedRayCoeffMarker) < 1e-6f) {
		double d2 = d * d;
		if (std::abs(cz[3] - kInvDepthCoeffMarker) < 1e-6f) {
			double inv_z = cz[0] + cz[1] * d + cz[2] * d2;
			if (!std::isfinite(inv_z) || std::abs(inv_z) < 1e-12) return false;
			z_rect = 1.0 / inv_z;
		}
		else {
			z_rect = cz[0] + cz[1] * d + cz[2] * d2 + cz[3] * d2 * d;
		}
		x_rect = cx[0] * z_rect;
		y_rect = cy[0] * z_rect;
		return std::isfinite(x_rect) && std::isfinite(y_rect) && std::isfinite(z_rect);
	}

	if (std::abs(cx[3] - kZLinearXYCoeffMarker) < 1e-6f) {
		if (std::abs(cz[3] - kRationalDepthCoeffMarker) < 1e-6f) {
			double denom = cz[1] * d + 1.0;
			if (!std::isfinite(denom) || std::abs(denom) < 1e-12) return false;
			z_rect = cz[0] / denom;
		}
		else {
			double d2 = d * d;
			double d3 = d2 * d;
			z_rect = cz[0] + cz[1] * d + cz[2] * d2 + cz[3] * d3;
		}
		x_rect = cx[0] * z_rect + cx[1];
		y_rect = cy[0] * z_rect + cy[1];
		return std::isfinite(x_rect) && std::isfinite(y_rect) && std::isfinite(z_rect);
	}
	if (std::abs(cz[3] - kInvDepthCoeffMarker) < 1e-6f) {
		double inv_z = cz[0] + cz[1] * d + cz[2] * d * d;
		if (!std::isfinite(inv_z) || std::abs(inv_z) < 1e-12) return false;
		z_rect = 1.0 / inv_z;
		x_rect = cx[0] * z_rect;
		y_rect = cy[0] * z_rect;
		return std::isfinite(x_rect) && std::isfinite(y_rect) && std::isfinite(z_rect);
	}

	double d2 = d * d;
	double d3 = d2 * d;
	x_rect = cx[0] + cx[1] * d + cx[2] * d2 + cx[3] * d3;
	y_rect = cy[0] + cy[1] * d + cy[2] * d2 + cy[3] * d3;
	z_rect = cz[0] + cz[1] * d + cz[2] * d2 + cz[3] * d3;
	return std::isfinite(x_rect) && std::isfinite(y_rect) && std::isfinite(z_rect);
}
}

void applyGaussianFilterToPointCloudV2(const cv::Mat& src_xyz, const cv::Mat& P,
                                        cv::Mat& dst_xyz, const int win_size)
{
	const float count_threshold = 0.8f;
	const int half_win_size = (win_size - 1) / 2;

	cv::Mat kerner1d = cv::getGaussianKernel(win_size, 0.7, CV_32FC1);
	cv::Mat kernel   = kerner1d * kerner1d.t();
	dst_xyz = cv::Mat::zeros(src_xyz.size(), CV_32FC3);

	#pragma omp parallel for collapse(2)
	for (int irow = half_win_size; irow < src_xyz.rows - half_win_size; irow++)
	{
		const cv::Vec3f* xyz_ptr   = src_xyz.ptr<cv::Vec3f>(irow);
		cv::Vec3f*       dst_xyz_ptr = dst_xyz.ptr<cv::Vec3f>(irow);
		const uchar*     P_ptr     = P.ptr<uchar>(irow);

		for (int jcol = half_win_size; jcol < src_xyz.cols - half_win_size; jcol++)
		{
			if (P_ptr[jcol] == 0)
				continue;

			cv::Vec3f sum(0, 0, 0);
			float count = 0;
			for (int ik = -half_win_size; ik <= half_win_size; ik++)
			{
				const cv::Vec3f* xyz_ptr0  = src_xyz.ptr<cv::Vec3f>(irow + ik);
				const uchar*     P_ptr0    = P.ptr<uchar>(irow + ik);
				const float*     kernel_ptr = kernel.ptr<float>(ik + half_win_size);
				for (int jk = -half_win_size; jk <= half_win_size; jk++)
				{
					if (P_ptr0[jcol + jk] == 0)
						continue;
					if (abs(xyz_ptr0[jcol + jk][0] - xyz_ptr[jcol][0]) < 3.0)
					{
						sum   += xyz_ptr0[jcol + jk] * kernel_ptr[half_win_size + jk];
						count += kernel_ptr[half_win_size + jk];
					}
				}
			}

			if (count > count_threshold)
				dst_xyz_ptr[jcol] = sum / count;
		}
	}
}

void applyRadiusOutlierFilterToPointCloud(const cv::Mat& src_xyz, const cv::Mat& P,
                                           cv::Mat& dst_xyz, cv::Mat& dst_P,
                                           const float search_radius, const int min_neighbors,
                                           const int window_size = 15)
{
	dst_xyz = cv::Mat::zeros(src_xyz.size(), CV_32FC3);
	dst_P   = cv::Mat::zeros(P.size(), CV_8UC1);

	const float radius_squared = search_radius * search_radius;
	const int half_win_size    = window_size / 2;

	#pragma omp parallel for collapse(2)
	for (int irow = 0; irow < src_xyz.rows; irow++)
	{
		for (int jcol = 0; jcol < src_xyz.cols; jcol++)
		{
			const uchar* P_ptr = P.ptr<uchar>(irow);

			if (P_ptr[jcol] == 0)
				continue;

			const cv::Vec3f* xyz_ptr       = src_xyz.ptr<cv::Vec3f>(irow);
			const cv::Vec3f& current_point = xyz_ptr[jcol];

			int neighbor_count = 0;

			int row_start = std::max(0, irow - half_win_size);
			int row_end   = std::min(src_xyz.rows, irow + half_win_size + 1);
			int col_start = std::max(0, jcol - half_win_size);
			int col_end   = std::min(src_xyz.cols, jcol + half_win_size + 1);

			for (int i = row_start; i < row_end; i++)
			{
				const cv::Vec3f* xyz_ptr_i = src_xyz.ptr<cv::Vec3f>(i);
				const uchar*     P_ptr_i   = P.ptr<uchar>(i);

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
				uchar*     dst_P_ptr   = dst_P.ptr<uchar>(irow);
				dst_xyz_ptr[jcol] = current_point;
				dst_P_ptr[jcol]   = 1;
			}
		}
	}
}

JSLAB_PhaseEpipolarReconstructor_V2::JSLAB_PhaseEpipolarReconstructor_V2(const Config& config) :
	m_config(config),
	m_is_init(false),
	m_min_disparity(0),
	m_max_disparity(0),
	m_pha_f(0)
{

}

JSLAB_PhaseEpipolarReconstructor_V2::~JSLAB_PhaseEpipolarReconstructor_V2()
{

}

void JSLAB_PhaseEpipolarReconstructor_V2::estimateDisparityRange()
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

static void computeFullLeftRectification(const cv::Mat& K1, const cv::Mat& D1,const cv::Size& image_size,
										 const cv::Mat& R1,cv::Mat& P1,cv::Mat& P2,cv::Mat& Q,cv::Size& rectified_size)
{
	std::vector<cv::Point2f> corners = {
		{0.0f, 0.0f},
		{(float)image_size.width - 1.0f, 0.0f},
		{(float)image_size.width - 1.0f, (float)image_size.height - 1.0f},
		{0.0f, (float)image_size.height - 1.0f}
	};

	std::vector<cv::Point2f> rect_corners;
	cv::undistortPoints(corners, rect_corners, K1, D1, R1, P1);

	float min_x = rect_corners[0].x;
	float max_x = rect_corners[0].x;
	float min_y = rect_corners[0].y;
	float max_y = rect_corners[0].y;

	for (const auto& p : rect_corners)
	{
		min_x = std::min(min_x, p.x);
		max_x = std::max(max_x, p.x);
		min_y = std::min(min_y, p.y);
		max_y = std::max(max_y, p.y);
	}

	double shift_x = -std::floor(min_x);
	double shift_y = -std::floor(min_y);

	rectified_size = cv::Size(
		(int)std::ceil(max_x - min_x + 1.0f),
		(int)std::ceil(max_y - min_y + 1.0f)
	);

	P1.at<double>(0, 2) += shift_x;
	P1.at<double>(1, 2) += shift_y;
	P2.at<double>(0, 2) += shift_x;
	P2.at<double>(1, 2) += shift_y;

	Q.at<double>(0, 3) -= shift_x;
	Q.at<double>(1, 3) -= shift_y;
}


void JSLAB_PhaseEpipolarReconstructor_V2::setCalibrationParams(const JSLAB_StereoCalibParams_V1& params)
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

	cv::Size output_size = m_params.image_size;
	bool use_full_left = m_config.use_full_left_rectification;
	cv::Size coeff_size;
	if (!m_params_v2.xyz_calib_coeffs[0].empty()) {
		coeff_size = m_params_v2.xyz_calib_coeffs[0].size();
		if (coeff_size != m_params.image_size) use_full_left = true;
	}
	if (use_full_left) {
		cv::Size full_rectified_size;
		computeFullLeftRectification(m_params.camera_matrix[0], m_params.dist_coeffs[0],
			m_params.image_size, m_R1, P1, P2, m_Q, full_rectified_size);
		output_size = full_rectified_size;
		if (!coeff_size.empty()) {
			output_size = coeff_size;
			if (full_rectified_size != coeff_size) {
				std::cout << "[V2 Warning] Coeff size " << coeff_size.width << "x" << coeff_size.height
					<< " differs from computed full rectification size "
					<< full_rectified_size.width << "x" << full_rectified_size.height
					<< ". Use coeff size for remap output." << std::endl;
			}
		}
		std::cout << "[V2] Full-left rectification output size: "
			<< output_size.width << "x" << output_size.height << std::endl;
	}
	cv::initUndistortRectifyMap(m_params.camera_matrix[0], m_params.dist_coeffs[0],
	                            m_R1, P1, output_size, CV_32FC1,
	                            m_left_remap[0], m_left_remap[1]);
	cv::initUndistortRectifyMap(m_params.camera_matrix[1], m_params.dist_coeffs[1],
	                            R2, P2, output_size, CV_32FC1,
	                            m_right_remap[0], m_right_remap[1]);
}
void JSLAB_PhaseEpipolarReconstructor_V2::setCalibrationParamsV2(const JSLAB_StereoCalibParams_V2& params)
{
	m_params_v2 = params;

	// 提取出 V1 的基础几何参数，复用原有的对极校正映射逻辑
	JSLAB_StereoCalibParams_V1 v1_params;
	params.camera_matrix[0].copyTo(v1_params.camera_matrix[0]);
	params.camera_matrix[1].copyTo(v1_params.camera_matrix[1]);
	params.dist_coeffs[0].copyTo(v1_params.dist_coeffs[0]);
	params.dist_coeffs[1].copyTo(v1_params.dist_coeffs[1]);
	params.R.copyTo(v1_params.R);
	params.t.copyTo(v1_params.t);
	v1_params.image_size = params.image_size;

	setCalibrationParams(v1_params);

	// 检查是否成功加载了 V2 独有的多项式矩阵
	if (!m_params_v2.xyz_calib_coeffs[0].empty() &&
		!m_params_v2.xyz_calib_coeffs[1].empty() &&
		!m_params_v2.xyz_calib_coeffs[2].empty())
	{
		m_use_v2_coeffs = true;
	}
	else {
		m_use_v2_coeffs = false;
	}
}

void JSLAB_PhaseEpipolarReconstructor_V2::setData(const cv::Mat& left, const cv::Mat& right,
                                                   const cv::Mat& left_k, const cv::Mat& right_k,
                                                   const cv::Mat& left_mod, int pha_f)
{
	cv::remap(left,  m_left,  m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);
	cv::remap(right, m_right, m_right_remap[0], m_right_remap[1], cv::INTER_LINEAR);

	cv::remap(left_k,  m_left_k,  m_left_remap[0], m_left_remap[1], cv::INTER_NEAREST);
	cv::remap(right_k, m_right_k, m_right_remap[0], m_right_remap[1], cv::INTER_NEAREST);

	cv::remap(left_mod, m_left_mod, m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);


	m_pha_f = pha_f;

	const cv::Size output_size = m_left.size();
	if (!m_is_init || m_xyz.size() != output_size || m_P.size() != output_size)
	{
		m_xyz     = cv::Mat(output_size, CV_32FC3, cv::Scalar::all(0.0));
		m_P       = cv::Mat(output_size, CV_8UC1,  cv::Scalar(0));
		m_is_init = true;
	}

	m_xyz = cv::Vec3f(0, 0, 0);
	m_P   = 0;
	// New input data invalidates the cached disparity used by ROI detection.
	m_dispmap.release();
}

void JSLAB_PhaseEpipolarReconstructor_V2::startCalculate(cv::Mat& xyz, cv::Mat& P)
{
	cv::Mat disp_map;

	stereoMatch(m_left, m_right, m_left_k, m_right_k, m_pha_f, m_dispmap);

	disp_map = m_dispmap;

	if (m_config.use_disp_median_filter)
	{
		disp_map.convertTo(disp_map, CV_32FC1);
		cv::medianBlur(disp_map, disp_map, 5);
		disp_map.convertTo(disp_map, CV_64FC1);
	}
	// Keep the diagnostic disparity map aligned with the filtered map used for reprojection.
	m_dispmap = disp_map.clone();

#pragma omp parallel for
	for (int irow = 0; irow < disp_map.rows; irow++)
	{
		const double* disp_ptr = disp_map.ptr<double>(irow);
		cv::Vec3f*    xyz_ptr  = m_xyz.ptr<cv::Vec3f>(irow);
		uchar*        P_ptr    = m_P.ptr<uchar>(irow);

		for (int jcol = 0; jcol < disp_map.cols; jcol++)
		{
			double d = disp_ptr[jcol];

			if (std::abs(d) < 1e-5)
				continue;

			bool v2_applied = false;

			double w = m_Q.at<double>(3, 2) * d + m_Q.at<double>(3, 3);
			double x = jcol + m_Q.at<double>(0, 3);
			double y = irow + m_Q.at<double>(1, 3);
			double z = m_Q.at<double>(2, 3);

			double inv_w = 1.0 / w;
			x *= inv_w;
			y *= inv_w;
			z *= inv_w;

			double rx = m_R1.at<double>(0, 0) * x + m_R1.at<double>(1, 0) * y + m_R1.at<double>(2, 0) * z;
			double ry = m_R1.at<double>(0, 1) * x + m_R1.at<double>(1, 1) * y + m_R1.at<double>(2, 1) * z;
			double rz = m_R1.at<double>(0, 2) * x + m_R1.at<double>(1, 2) * y + m_R1.at<double>(2, 2) * z;

			if (m_config.use_rectified_coords)
			{
				if (z > m_config.min_depth && z < m_config.max_depth)
				{
					xyz_ptr[jcol] = cv::Vec3f((float)x, (float)y, (float)z);
					P_ptr[jcol]   = 1;
				}
			}
			else
			{
				if (rz > m_config.min_depth && rz < m_config.max_depth)
				{
					xyz_ptr[jcol] = cv::Vec3f((float)rx, (float)ry, (float)rz);
					P_ptr[jcol]   = 1;
				}
			}
		}
	}

	applyGaussianFilterToPointCloudV2(m_xyz, m_P, xyz, 3);

	cv::Mat xyz_filtered, P_filtered;
	applyRadiusOutlierFilterToPointCloud(xyz, m_P, xyz_filtered, P_filtered,
		m_config.radius_search, m_config.radius_min_neighbors, m_config.radius_window_size);
	xyz = xyz_filtered;
	P   = P_filtered;

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

void JSLAB_PhaseEpipolarReconstructor_V2::startCalculate_v2(cv::Mat& xyz, cv::Mat& P)
{
	stereoMatch(m_left, m_right, m_left_k, m_right_k, m_pha_f, m_dispmap);


	if (m_config.use_disp_median_filter) {
		m_dispmap.convertTo(m_dispmap, CV_32FC1);
		cv::medianBlur(m_dispmap, m_dispmap, 5);
		m_dispmap.convertTo(m_dispmap, CV_64FC1);
	}
	cv::Mat disp_map = m_dispmap;
#pragma omp parallel for
	for (int irow = 0; irow < disp_map.rows; irow++)
	{
		const double* disp_ptr = disp_map.ptr<double>(irow);
		cv::Vec3f* xyz_ptr = m_xyz.ptr<cv::Vec3f>(irow);
		uchar* P_ptr = m_P.ptr<uchar>(irow);

		for (int jcol = 0; jcol < disp_map.cols; jcol++)
		{
			double d = disp_ptr[jcol];
			if (std::abs(d) < 1e-5) continue;

			bool processed = false;

			// =========================================================
			// 【路径 A】：V2 三次方程求解 (此时输出的为极线校正后坐标)
			// =========================================================
			if (m_use_v2_coeffs)
			{
				// 读取 4 通道系数 (Vec4f)
				cv::Vec4f cz = m_params_v2.xyz_calib_coeffs[2].at<cv::Vec4f>(irow, jcol);

				if (std::abs(cz[0]) > 1e-9 || std::abs(cz[1]) > 1e-9 || std::abs(cz[2]) > 1e-9 || std::abs(cz[3]) > 1e-9)
				{
					cv::Vec4f cx = m_params_v2.xyz_calib_coeffs[0].at<cv::Vec4f>(irow, jcol);
					cv::Vec4f cy = m_params_v2.xyz_calib_coeffs[1].at<cv::Vec4f>(irow, jcol);

					double x_rect = 0.0, y_rect = 0.0, z_rect = 0.0;
					if (!evaluateV2CalibPoint(cx, cy, cz, d, x_rect, y_rect, z_rect)) continue;

					// 须用 R1 矩阵旋转回原相机的物理坐标系
					double rx = m_R1.at<double>(0, 0) * x_rect + m_R1.at<double>(1, 0) * y_rect + m_R1.at<double>(2, 0) * z_rect;
					double ry = m_R1.at<double>(0, 1) * x_rect + m_R1.at<double>(1, 1) * y_rect + m_R1.at<double>(2, 1) * z_rect;
					double rz = m_R1.at<double>(0, 2) * x_rect + m_R1.at<double>(1, 2) * y_rect + m_R1.at<double>(2, 2) * z_rect;

					if (m_config.use_rectified_coords)
					{
						if (z_rect > m_config.min_depth && z_rect < m_config.max_depth)
						{
							xyz_ptr[jcol] = cv::Vec3f((float)x_rect, (float)y_rect, (float)z_rect);
							P_ptr[jcol] = 1;
						}
					}
					else
					{
						if (rz > m_config.min_depth && rz < m_config.max_depth)
						{
							xyz_ptr[jcol] = cv::Vec3f((float)rx, (float)ry, (float)rz);
							P_ptr[jcol] = 1;
						}
						processed = true;
					}
				}
			}

			// =========================================================
			// 【路径 B】：传统保底回退 (Fallback)
			// =========================================================
			//if (!processed)
			//{
			//	double w = m_Q.at<double>(3, 2) * d + m_Q.at<double>(3, 3);
			//	double x = (jcol + m_Q.at<double>(0, 3)) / w;
			//	double y = (irow + m_Q.at<double>(1, 3)) / w;
			//	double z = m_Q.at<double>(2, 3) / w;

			//	double rx = m_R1.at<double>(0, 0) * x + m_R1.at<double>(1, 0) * y + m_R1.at<double>(2, 0) * z;
			//	double ry = m_R1.at<double>(0, 1) * x + m_R1.at<double>(1, 1) * y + m_R1.at<double>(2, 1) * z;
			//	double rz = m_R1.at<double>(0, 2) * x + m_R1.at<double>(1, 2) * y + m_R1.at<double>(2, 2) * z;

			//	if (m_config.use_rectified_coords)
			//	{
			//		if (z > m_config.min_depth && z < m_config.max_depth)
			//		{
			//			xyz_ptr[jcol] = cv::Vec3f((float)x, (float)y, (float)z);
			//			P_ptr[jcol] = 1;
			//		}
			//	}
			//	else
			//	{
			//		if (rz > m_config.min_depth && rz < m_config.max_depth)
			//		{
			//			xyz_ptr[jcol] = cv::Vec3f((float)rx, (float)ry, (float)rz);
			//			P_ptr[jcol] = 1;
			//		}
			//	}
			//}
		}
	}

	// 4. 后处理流程
	applyGaussianFilterToPointCloudV2(m_xyz, m_P, xyz, 3);
	cv::Mat xyz_filtered, P_filtered;
	applyRadiusOutlierFilterToPointCloud(xyz, m_P, xyz_filtered, P_filtered, m_config.radius_search, m_config.radius_min_neighbors, m_config.radius_window_size);
	xyz = xyz_filtered;
	P = P_filtered;
	if (m_config.use_mls) {
		cv::Mat xyz_mls;
		applyMLSFilterToPointCloud(xyz, P, xyz_mls, m_config.mls_window_size, m_config.mls_spatial_sigma, m_config.use_gpu_mls, m_config.mls_polynomial_order, m_config.mls_max_radius, m_config.mls_iterations);
		xyz = xyz_mls;
	}
}

bool JSLAB_PhaseEpipolarReconstructor_V2::prepareRectifiedROI(const cv::RotatedRect& roi_original, RectifiedROI& roi) const
{
	roi.left_rect = cv::Rect();
	roi.right_rect = cv::Rect();
	roi.mask.release();

	if (m_left.empty() || m_left_remap[0].empty() || m_left_remap[1].empty()) return false;

	cv::Mat original_mask = cv::Mat::zeros(m_params.image_size, CV_8UC1);
	cv::Point2f pts_f[4];
	roi_original.points(pts_f);
	cv::Point pts_i[4];
	for (int i = 0; i < 4; ++i) {
		pts_i[i] = cv::Point(cvRound(pts_f[i].x), cvRound(pts_f[i].y));
	}
	cv::fillConvexPoly(original_mask, pts_i, 4, cv::Scalar(255));

	cv::Mat rect_mask_linear, rect_mask;
	cv::remap(original_mask, rect_mask_linear, m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);
	cv::threshold(rect_mask_linear, rect_mask, 0, 255, cv::THRESH_BINARY);

	// Match V8's "mask before remap" behavior more closely: linear remap spreads valid
	// edge pixels, and the 5x5 median disparity filter needs a small valid border.
	cv::Mat roi_edge_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::dilate(rect_mask, rect_mask, roi_edge_kernel);

	std::vector<cv::Point> valid_pts;
	cv::findNonZero(rect_mask, valid_pts);
	if (valid_pts.empty()) return false;

	cv::Rect image_rect(0, 0, m_left.cols, m_left.rows);
	cv::Rect left_rect = cv::boundingRect(valid_pts) & image_rect;
	if (left_rect.empty()) return false;

	// Build the right search window from the fringe orders that actually appear in the left ROI.
	// This avoids both failure modes: a narrow disparity crop can miss order 38 near image borders,
	// while a full-row crop is unnecessarily expensive.
	std::vector<uchar> needed_orders(std::max(m_pha_f + 1, 1), 0);
	for (int y = 0; y < left_rect.height; ++y) {
		const uchar* mptr = rect_mask.ptr<uchar>(left_rect.y + y) + left_rect.x;
		const int* kptr = m_left_k.ptr<int>(left_rect.y + y) + left_rect.x;
		for (int x = 0; x < left_rect.width; ++x) {
			if (mptr[x] == 0) continue;
			int k = kptr[x];
			if (k > 0 && k <= m_pha_f) needed_orders[k] = 1;
		}
	}

	int right_min_x = m_right.cols;
	int right_max_x = -1;
	for (int y = left_rect.y; y < left_rect.y + left_rect.height; ++y) {
		if (y < 0 || y >= m_right_k.rows) continue;
		const int* rptr = m_right_k.ptr<int>(y);
		for (int x = 0; x < m_right_k.cols; ++x) {
			int k = rptr[x];
			if (k > 0 && k <= m_pha_f && needed_orders[k]) {
				right_min_x = std::min(right_min_x, x);
				right_max_x = std::max(right_max_x, x);
			}
		}
	}

	const int order_search_pad = 32;
	cv::Rect right_rect;
	if (right_max_x >= right_min_x) {
		int x0 = std::max(0, right_min_x - order_search_pad);
		int x1 = std::min(m_right.cols - 1, right_max_x + order_search_pad);
		right_rect = cv::Rect(x0, left_rect.y, x1 - x0 + 1, left_rect.height);
	}
	else {
		// Fallback for abnormal order maps: keep correctness over speed.
		right_rect = cv::Rect(0, left_rect.y, m_right.cols, left_rect.height);
	}
	right_rect &= image_rect;
	if (right_rect.empty()) return false;

	roi.left_rect = left_rect;
	roi.right_rect = right_rect;
	roi.mask = rect_mask(left_rect).clone();
	return cv::countNonZero(roi.mask) > 0;
}

void JSLAB_PhaseEpipolarReconstructor_V2::startCalculate_v2_roi(cv::Mat& xyz, cv::Mat& P, const cv::RotatedRect& roi_original)
{
	RectifiedROI roi;
	if (!prepareRectifiedROI(roi_original, roi)) {
		xyz.release();
		P.release();
		return;
	}
	startCalculate_v2_roi(xyz, P, roi);
}

void JSLAB_PhaseEpipolarReconstructor_V2::startCalculate_v2_roi(cv::Mat& xyz, cv::Mat& P, const RectifiedROI& roi)
{
	if (!m_use_v2_coeffs || m_left.empty() || m_right.empty()) {
		startCalculate_v2(xyz, P);
		return;
	}
	if (roi.left_rect.empty() || roi.right_rect.empty() || roi.mask.empty()) {
		xyz.release();
		P.release();
		return;
	}

	const cv::Rect& left_rect = roi.left_rect;
	const cv::Rect& right_rect = roi.right_rect;
	const cv::Mat& roi_mask = roi.mask;

	cv::Mat l_phase = m_left(left_rect);
	cv::Mat l_k = m_left_k(left_rect);
	cv::Mat r_phase = m_right(right_rect);
	cv::Mat r_k = m_right_k(right_rect);

	cv::Mat disp_map = cv::Mat::zeros(left_rect.size(), CV_64FC1);
	const int max_rank_limit = m_pha_f;
	const int knum = m_pha_f;

	long long roi_mask_count = 0;
	long long rank_valid_count = 0;
	long long rank_empty_count = 0;
	long long cost_reject_count = 0;
	long long edge_reject_count = 0;
	long long disp_valid_count = 0;

#pragma omp parallel for reduction(+:roi_mask_count, rank_valid_count, rank_empty_count, cost_reject_count, edge_reject_count, disp_valid_count)
	for (int irow = 0; irow < l_phase.rows; ++irow)
	{
		const double* l_phase_ptr = l_phase.ptr<double>(irow);
		const double* r_phase_ptr = r_phase.ptr<double>(irow);
		const int* l_k_ptr = l_k.ptr<int>(irow);
		const int* r_k_ptr = r_k.ptr<int>(irow);
		const uchar* mask_ptr = roi_mask.ptr<uchar>(irow);
		double* disp_ptr = disp_map.ptr<double>(irow);

		std::vector<std::vector<int>> row_rank_table(knum + 1);
		for (int j = 0; j < r_phase.cols; ++j) {
			int k_val = r_k_ptr[j];
			if (k_val >= 0 && k_val <= knum) row_rank_table[k_val].push_back(j);
		}

		for (int j = 0; j < l_phase.cols; ++j) {
			if (mask_ptr[j] == 0) continue;
			roi_mask_count++;

			int k = l_k_ptr[j];
			if (k > max_rank_limit || k <= 0 || k > knum) continue;
			rank_valid_count++;
			if (row_rank_table[k].empty()) {
				rank_empty_count++;
				continue;
			}

			double min_cost = 1e9;
			int best_jR = -1;
			double l_val = l_phase_ptr[j];
			const std::vector<int>& candidates = row_rank_table[k];

			for (int cand_idx : candidates) {
				double diff = std::abs(r_phase_ptr[cand_idx] - l_val);
				if (diff < min_cost) {
					min_cost = diff;
					best_jR = cand_idx;
				}
			}

			if (min_cost >= m_config.grad_threshold) {
				cost_reject_count++;
				continue;
			}
			if (best_jR < 2 || best_jR >= r_phase.cols - 1) {
				edge_reject_count++;
				continue;
			}

			double cost0 = r_phase_ptr[best_jR - 1] - l_val;
			double cost1 = r_phase_ptr[best_jR] - l_val;
			double cost2 = r_phase_ptr[best_jR + 1] - l_val;
			double numeratorL = cost1 * (cost0 - cost2);
			double denominatorL = 2.0 * (cost0 - cost1) * (cost1 - cost2);
			double delta_L = 0.0;
			if (std::abs(denominatorL) > 1e-9) delta_L = numeratorL / denominatorL;

			double global_left_x = static_cast<double>(left_rect.x + j);
			double global_right_x = static_cast<double>(right_rect.x + best_jR);
			disp_ptr[j] = global_left_x - global_right_x - delta_L;
			disp_valid_count++;
		}
	}

	std::cout << "[V2 ROI Debug] rectified_left=" << left_rect << ", rectified_right=" << right_rect
		<< ", roi_mask=" << roi_mask_count
		<< ", rank_valid=" << rank_valid_count
		<< ", rank_empty=" << rank_empty_count
		<< ", cost_reject=" << cost_reject_count
		<< ", edge_reject=" << edge_reject_count
		<< ", disp_valid=" << disp_valid_count
		<< ", grad_threshold=" << m_config.grad_threshold << std::endl;

	if (m_config.use_disp_median_filter) {
		disp_map.convertTo(disp_map, CV_32FC1);
		cv::medianBlur(disp_map, disp_map, 5);
		disp_map.convertTo(disp_map, CV_64FC1);
	}

	cv::Mat xyz_raw = cv::Mat::zeros(left_rect.size(), CV_32FC3);
	cv::Mat P_raw = cv::Mat::zeros(left_rect.size(), CV_8UC1);

	long long reproj_mask_count = 0;
	long long reproj_disp_zero_count = 0;
	long long reproj_coeff_zero_count = 0;
	long long reproj_eval_fail_count = 0;
	long long reproj_depth_reject_count = 0;
	long long reproj_valid_count = 0;
	double min_eval_z = std::numeric_limits<double>::max();
	double max_eval_z = -std::numeric_limits<double>::max();
	double min_pass_z = std::numeric_limits<double>::max();
	double max_pass_z = -std::numeric_limits<double>::max();

#pragma omp parallel for reduction(+:reproj_mask_count, reproj_disp_zero_count, reproj_coeff_zero_count, reproj_eval_fail_count, reproj_depth_reject_count, reproj_valid_count)
	for (int irow = 0; irow < disp_map.rows; ++irow)
	{
		const double* disp_ptr = disp_map.ptr<double>(irow);
		const uchar* mask_ptr = roi_mask.ptr<uchar>(irow);
		cv::Vec3f* xyz_ptr = xyz_raw.ptr<cv::Vec3f>(irow);
		uchar* P_ptr = P_raw.ptr<uchar>(irow);
		int global_y = left_rect.y + irow;

		for (int jcol = 0; jcol < disp_map.cols; ++jcol) {
			if (mask_ptr[jcol] == 0) continue;
			reproj_mask_count++;
			double d = disp_ptr[jcol];
			if (std::abs(d) < 1e-5) {
				reproj_disp_zero_count++;
				continue;
			}

			int global_x = left_rect.x + jcol;
			cv::Vec4f cz = m_params_v2.xyz_calib_coeffs[2].at<cv::Vec4f>(global_y, global_x);
			if (std::abs(cz[0]) <= 1e-9 && std::abs(cz[1]) <= 1e-9 && std::abs(cz[2]) <= 1e-9 && std::abs(cz[3]) <= 1e-9) {
				reproj_coeff_zero_count++;
				continue;
			}

			cv::Vec4f cx = m_params_v2.xyz_calib_coeffs[0].at<cv::Vec4f>(global_y, global_x);
			cv::Vec4f cy = m_params_v2.xyz_calib_coeffs[1].at<cv::Vec4f>(global_y, global_x);

			double x_rect = 0.0, y_rect = 0.0, z_rect = 0.0;
			if (!evaluateV2CalibPoint(cx, cy, cz, d, x_rect, y_rect, z_rect)) {
				reproj_eval_fail_count++;
				continue;
			}
			min_eval_z = std::min(min_eval_z, z_rect);

			double rx = m_R1.at<double>(0, 0) * x_rect + m_R1.at<double>(1, 0) * y_rect + m_R1.at<double>(2, 0) * z_rect;
			double ry = m_R1.at<double>(0, 1) * x_rect + m_R1.at<double>(1, 1) * y_rect + m_R1.at<double>(2, 1) * z_rect;
			double rz = m_R1.at<double>(0, 2) * x_rect + m_R1.at<double>(1, 2) * y_rect + m_R1.at<double>(2, 2) * z_rect;

			bool depth_ok = false;
			if (m_config.use_rectified_coords) {
				depth_ok = (z_rect > m_config.min_depth && z_rect < m_config.max_depth);
				if (depth_ok) {
					xyz_ptr[jcol] = cv::Vec3f((float)x_rect, (float)y_rect, (float)z_rect);
					P_ptr[jcol] = 1;
					min_pass_z = std::min(min_pass_z, z_rect);
					max_pass_z = std::max(max_pass_z, z_rect);
					reproj_valid_count++;
				}
			}
			else {
				depth_ok = (rz > m_config.min_depth && rz < m_config.max_depth);
				if (depth_ok) {
					xyz_ptr[jcol] = cv::Vec3f((float)rx, (float)ry, (float)rz);
					P_ptr[jcol] = 1;
					min_pass_z = std::min(min_pass_z, rz);
					max_pass_z = std::max(max_pass_z, rz);
					reproj_valid_count++;
				}
			}
			if (!depth_ok) reproj_depth_reject_count++;
		}
	}

	std::cout << "[V2 ROI Debug] reproj_mask=" << reproj_mask_count
		<< ", disp_zero=" << reproj_disp_zero_count
		<< ", coeff_zero=" << reproj_coeff_zero_count
		<< ", eval_fail=" << reproj_eval_fail_count
		<< ", depth_reject=" << reproj_depth_reject_count
		<< ", raw_valid=" << reproj_valid_count
		<< ", depth_range=[" << m_config.min_depth << ", " << m_config.max_depth << "]"
		<< ", use_rectified=" << m_config.use_rectified_coords << std::endl;

	applyGaussianFilterToPointCloudV2(xyz_raw, P_raw, xyz, 3);
	cv::Mat xyz_filtered, P_filtered;
	applyRadiusOutlierFilterToPointCloud(xyz, P_raw, xyz_filtered, P_filtered,
		m_config.radius_search, m_config.radius_min_neighbors, m_config.radius_window_size);

	const int raw_valid = cv::countNonZero(P_raw);
	const int radius_valid = cv::countNonZero(P_filtered);
	const bool radius_too_aggressive = (raw_valid > 0 && radius_valid < raw_valid * 0.1);
	if (radius_too_aggressive) {
		P = P_raw.clone();
	}
	else {
		xyz = xyz_filtered;
		P = P_filtered;
	}

	std::cout << "[V2 ROI Debug] after_filters raw_valid=" << raw_valid
		<< ", radius_valid=" << radius_valid
		<< ", used_radius=" << (!radius_too_aggressive) << std::endl;

	if (m_config.use_mls) {
		cv::Mat xyz_mls;
		applyMLSFilterToPointCloud(xyz, P, xyz_mls,
			m_config.mls_window_size, m_config.mls_spatial_sigma, m_config.use_gpu_mls,
			m_config.mls_polynomial_order, m_config.mls_max_radius, m_config.mls_iterations);
		xyz = xyz_mls;
	}
}
void JSLAB_PhaseEpipolarReconstructor_V2::getDisparityMap(cv::Mat& disp) const
{
	disp = m_dispmap.clone();
}

void JSLAB_PhaseEpipolarReconstructor_V2::rectifyLeftImage(const cv::Mat& image, cv::Mat& rectified) const
{
	if (image.empty() || m_left_remap[0].empty() || m_left_remap[1].empty()) { rectified.release(); return; }
	cv::remap(image, rectified, m_left_remap[0], m_left_remap[1], cv::INTER_LINEAR);
}

void JSLAB_PhaseEpipolarReconstructor_V2::getRectifiedLeftModulation(cv::Mat& mod) const
{
	mod = m_left_mod.clone();
}



void JSLAB_PhaseEpipolarReconstructor_V2::stereoMatch(const cv::Mat& l_phase, const cv::Mat& r_phase,
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
		const int*    l_k_ptr     = l_k.ptr<int>(i);
		const int*    r_k_ptr     = r_k.ptr<int>(i);
		double*       disp_ptr    = disp.ptr<double>(i);

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
			int    best_jR  = -1;
			double l_val    = l_phase_ptr[j];
			const std::vector<int>& candidates = row_rank_table[k];

			for (int cand_idx : candidates)
			{
				double diff = std::abs(r_phase_ptr[cand_idx] - l_val);
				if (diff < min_cost)
				{
					min_cost = diff;
					best_jR  = cand_idx;
				}
			}

			if (min_cost >= m_config.grad_threshold)
				continue;

			if (best_jR < 2 || best_jR >= r_phase.cols - 1)
				continue;

			double cost0 = r_phase_ptr[best_jR - 1] - l_val;
			double cost1 = r_phase_ptr[best_jR]     - l_val;
			double cost2 = r_phase_ptr[best_jR + 1] - l_val;

			double numeratorL   = cost1 * (cost0 - cost2);
			double denominatorL = 2.0 * (cost0 - cost1) * (cost1 - cost2);

			double delta_L = 0.0;
			if (std::abs(denominatorL) > 1e-9)
				delta_L = numeratorL / denominatorL;

			disp_ptr[j] = (double)j - (double)best_jR - delta_L;
		}
	}
}


// =========================================================================
// 辅助鼠标回调：用于在图像上点击选取 N 个自适应阶梯块种子点
// =========================================================================
struct PointClickData {
	std::vector<cv::Point> pts;
	int max_pts = 2; // 需要点击的总次数
	int revision = 0;
};

static void onMouseClick(int event, int x, int y, int flags, void* userdata) {
	PointClickData* data = reinterpret_cast<PointClickData*>(userdata);
	if (event == cv::EVENT_LBUTTONDOWN) {
		if (data->pts.size() < data->max_pts) {
			data->pts.push_back(cv::Point(x, y));
			++data->revision;
		}
	}
	else if (event == cv::EVENT_RBUTTONDOWN && !data->pts.empty()) {
		data->pts.pop_back();
		++data->revision;
	}
}
// =========================================================================

std::vector<cv::RotatedRect> JSLAB_PhaseEpipolarReconstructor_V2::detectTargetROIs(int keep_num, int min_area_thresh, float grad_thresh)
{
	std::vector<cv::RotatedRect> found_rois;
	std::cout << ">>> Running ROI detection (robust gradient + seed flood fill) ..." << std::endl;

	if (m_dispmap.empty()) {
		stereoMatch(m_left, m_right, m_left_k, m_right_k, m_pha_f, m_dispmap);
	}

	cv::Mat disp_32f;
	m_dispmap.convertTo(disp_32f, CV_32FC1);
	cv::patchNaNs(disp_32f, 0.0f);

	cv::Mat disp_smooth;
	cv::medianBlur(disp_32f, disp_smooth, 3);

	cv::Mat grad_x, grad_y, abs_x, abs_y, grad_mag;
	//cv::Scharr(disp_smooth, grad_x, CV_32FC1, 1, 0, 1.0, 0.0, cv::BORDER_REPLICATE);
	cv::Sobel(disp_smooth, grad_x, CV_32FC1, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);

	cv::Sobel(disp_smooth, grad_y, CV_32FC1, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
	cv::absdiff(grad_x, cv::Scalar(0), abs_x);
	cv::absdiff(grad_y, cv::Scalar(0), abs_y);
	cv::addWeighted(abs_x, 1.0, abs_y, 1.00, 0.0, grad_mag);

	// A finite difference across a wider x gap makes shallow step edges more visible than a 1-pixel derivative.
	const int jump_gap = 5;
	cv::Mat jump = cv::Mat::zeros(disp_smooth.size(), CV_32FC1);
	for (int y = 0; y < disp_smooth.rows; ++y) {
		const float* src = disp_smooth.ptr<float>(y);
		float* dst = jump.ptr<float>(y);
		for (int x = jump_gap; x < disp_smooth.cols - jump_gap; ++x) {
			float left_v = src[x - jump_gap];
			float right_v = src[x + jump_gap];
			if (std::isfinite(left_v) && std::isfinite(right_v)) {
				dst[x] = std::abs(right_v - left_v);
			}
		}
	}
	cv::medianBlur(jump, jump, 5);
	cv::Mat jump_scaled;
	jump.convertTo(jump_scaled, CV_32FC1, 4.0);
	cv::max(grad_mag, jump_scaled, grad_mag);
	cv::medianBlur(grad_mag, grad_mag, 3);

	auto makeFlatMask = [&](float flat_thresh, int open_size) {
		cv::Mat mask = (grad_mag < flat_thresh);
		mask.convertTo(mask, CV_8UC1, 255);

		open_size = std::max(3, open_size | 1);
		cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(open_size, open_size));
		cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_kernel);
		return mask;
	};

	auto estimateSeedDisparity = [&](const cv::Point& pt, int half_win, float& seed_disp, float& disp_tol) {
		int x1 = std::max(0, pt.x - half_win);
		int y1 = std::max(0, pt.y - half_win);
		int x2 = std::min(disp_smooth.cols - 1, pt.x + half_win);
		int y2 = std::min(disp_smooth.rows - 1, pt.y + half_win);

		std::vector<float> vals;
		vals.reserve((x2 - x1 + 1) * (y2 - y1 + 1));
		for (int y = y1; y <= y2; ++y) {
			const float* ptr = disp_smooth.ptr<float>(y);
			for (int x = x1; x <= x2; ++x) {
				if (std::isfinite(ptr[x])) {
					vals.push_back(ptr[x]);
				}
			}
		}

		if (vals.empty()) {
			return false;
		}

		std::sort(vals.begin(), vals.end());
		seed_disp = vals[vals.size() / 2];

		std::vector<float> abs_dev;
		abs_dev.reserve(vals.size());
		for (float v : vals) {
			abs_dev.push_back(std::abs(v - seed_disp));
		}
		std::sort(abs_dev.begin(), abs_dev.end());
		float disp_mad = abs_dev[abs_dev.size() / 2];

		disp_tol = 8.0f * std::max(disp_mad, 0.05f) + 0.35f;
		disp_tol = std::max(0.75f, std::min(disp_tol, 6.0f));
		return true;
	};
	auto findNearestSeed = [](const cv::Mat& mask, const cv::Point& pt, int max_radius, cv::Point& seed) {
		if (pt.x >= 0 && pt.x < mask.cols && pt.y >= 0 && pt.y < mask.rows && mask.at<uchar>(pt) > 0) {
			seed = pt;
			return true;
		}

		for (int r = 1; r <= max_radius; ++r) {
			for (int dy = -r; dy <= r; ++dy) {
				for (int dx = -r; dx <= r; ++dx) {
					if (std::abs(dx) != r && std::abs(dy) != r) {
						continue;
					}
					cv::Point p(pt.x + dx, pt.y + dy);
					if (p.x >= 0 && p.x < mask.cols && p.y >= 0 && p.y < mask.rows && mask.at<uchar>(p) > 0) {
						seed = p;
						return true;
					}
				}
			}
		}
		return false;
	};

	auto mapRectToOriginal = [&](const cv::RotatedRect& rbox_rectified) {
		cv::Point2f pts_rect[4];
		rbox_rectified.points(pts_rect);

		std::vector<cv::Point2f> mapped_pts;
		mapped_pts.reserve(36);
		for (int j = 0; j < 4; ++j) {
			cv::Point2f a = pts_rect[j];
			cv::Point2f b = pts_rect[(j + 1) % 4];
			for (int s = 0; s <= 8; ++s) {
				float t = static_cast<float>(s) / 8.0f;
				cv::Point2f p = a + (b - a) * t;
				int rx = std::max(0, std::min(cvRound(p.x), m_left_remap[0].cols - 1));
				int ry = std::max(0, std::min(cvRound(p.y), m_left_remap[0].rows - 1));
				mapped_pts.push_back(cv::Point2f(m_left_remap[0].at<float>(ry, rx), m_left_remap[1].at<float>(ry, rx)));
			}
		}

		cv::RotatedRect rbox_original = cv::minAreaRect(mapped_pts);
		rbox_original.size.width *= 0.9f;
		rbox_original.size.height *= 0.9f;
		return rbox_original;
	};

	auto extractLargestRect = [&](const cv::Mat& target_mask, cv::RotatedRect& rbox) {
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(target_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		if (contours.empty()) {
			return false;
		}

		int max_idx = 0;
		double max_area = 0.0;
		for (size_t i = 0; i < contours.size(); ++i) {
			double area = cv::contourArea(contours[i]);
			if (area > max_area) {
				max_area = area;
				max_idx = static_cast<int>(i);
			}
		}

		if (max_area < std::max(50, min_area_thresh / 20)) {
			return false;
		}

		rbox = cv::minAreaRect(contours[max_idx]);
		return true;
	};

	if (grad_thresh < 0.0f)
	{
		std::cout << ">>> Adaptive mode: click " << keep_num << " target seeds." << std::endl;

		cv::Mat vis_base;
		if (!m_left_mod.empty()) {
			cv::normalize(m_left_mod, vis_base, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
			clahe->apply(vis_base, vis_base);
		}
		else {
			cv::Mat disp_clamped;
			cv::threshold(m_dispmap, disp_clamped, 0, 0, cv::THRESH_TOZERO);
			cv::normalize(disp_clamped, vis_base, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		}

		cv::Mat vis_color;
		cv::cvtColor(vis_base, vis_color, cv::COLOR_GRAY2BGR);

		double scale = 1200.0 / vis_color.rows;
		if (scale > 1.0) scale = 1.0;
		cv::Mat small_vis;
		cv::resize(vis_color, small_vis, cv::Size(), scale, scale);

		PointClickData click_data;
		click_data.max_pts = keep_num;

		std::string winName = "Click " + std::to_string(keep_num) + " Points on Targets";
		cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
		cv::setMouseCallback(winName, onMouseClick, &click_data);

		auto buildRoiFromClick = [&](const cv::Point& click_pt, int roi_index,
			cv::RotatedRect& rbox_rectified, cv::RotatedRect& rbox_original) {
			cv::Point orig_pt(cvRound(click_pt.x / scale), cvRound(click_pt.y / scale));
			orig_pt.x = std::max(0, std::min(orig_pt.x, grad_mag.cols - 1));
			orig_pt.y = std::max(0, std::min(orig_pt.y, grad_mag.rows - 1));

			const int half_win = 30;
			int x1 = std::max(0, orig_pt.x - half_win);
			int y1 = std::max(0, orig_pt.y - half_win);
			int x2 = std::min(grad_mag.cols - 1, orig_pt.x + half_win);
			int y2 = std::min(grad_mag.rows - 1, orig_pt.y + half_win);

			cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
			cv::Mat grad_roi = grad_mag(roi);

			std::vector<float> grad_vals;
			grad_vals.reserve(grad_roi.total());
			for (int r = 0; r < grad_roi.rows; ++r) {
				const float* ptr = grad_roi.ptr<float>(r);
				for (int c = 0; c < grad_roi.cols; ++c) {
					if (std::isfinite(ptr[c])) {
						grad_vals.push_back(ptr[c]);
					}
				}
			}

			if (grad_vals.empty()) {
				return false;
			}

			std::sort(grad_vals.begin(), grad_vals.end());
			float median = grad_vals[grad_vals.size() / 2];

			std::vector<float> abs_dev;
			abs_dev.reserve(grad_vals.size());
			for (float v : grad_vals) {
				abs_dev.push_back(std::abs(v - median));
			}
			std::sort(abs_dev.begin(), abs_dev.end());
			float mad = abs_dev[abs_dev.size() / 2];

			float p75 = grad_vals[std::min(grad_vals.size() - 1, static_cast<size_t>(grad_vals.size() * 75 / 100))];
			float robust_sigma = std::max(mad, 0.015f);
			float local_thresh = median + 2.4f * robust_sigma;
			local_thresh = std::max(local_thresh, p75 * 0.95f);
			local_thresh = std::max(0.04f, std::min(local_thresh, 4.5f));

			cv::Mat local_mask = makeFlatMask(local_thresh, 3);

			//float seed_disp = 0.0f;
			//float disp_tol = 0.0f;
			//if (estimateSeedDisparity(orig_pt, 9, seed_disp, disp_tol)) {
			//	cv::Mat disp_diff, disp_gate;
			//	cv::absdiff(disp_smooth, cv::Scalar(seed_disp), disp_diff);
			//	disp_gate = (disp_diff < disp_tol);
			//	disp_gate.convertTo(disp_gate, CV_8UC1, 255);
			//	cv::bitwise_and(local_mask, disp_gate, local_mask);
			//}

			cv::Mat final_open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
			cv::morphologyEx(local_mask, local_mask, cv::MORPH_OPEN, final_open_kernel);
			cv::Point seed;
			if (!findNearestSeed(local_mask, orig_pt, 25, seed)) {
				std::cout << ">>> ROI " << roi_index + 1 << ": no flat seed found." << std::endl;
				return false;
			}

			cv::Mat grow = local_mask.clone();
			cv::floodFill(grow, seed, cv::Scalar(128), nullptr, cv::Scalar(0), cv::Scalar(0), 4 | cv::FLOODFILL_FIXED_RANGE);
			cv::Mat target_cc = (grow == 128);

			if (!extractLargestRect(target_cc, rbox_rectified)) return false;
			rbox_original = mapRectToOriginal(rbox_rectified);
			rbox_rectified.size.width *= 0.9f;
			rbox_rectified.size.height *= 0.9f;
			return true;
		};

		std::vector<cv::RotatedRect> preview_rects;
		std::vector<cv::RotatedRect> original_rects;
		std::vector<uchar> valid_selections;
		int processed_revision = -1;
		bool confirmed = false;

		std::cout << ">>> Left click selects a target; right click removes the latest selection."
			<< std::endl;
		std::cout << ">>> Press Enter to confirm all ROI boxes, or Esc to cancel."
			<< std::endl;

		while (true) {
			if (processed_revision != click_data.revision) {
				preview_rects.resize(click_data.pts.size());
				original_rects.resize(click_data.pts.size());
				valid_selections.assign(click_data.pts.size(), 0);

				for (size_t i = 0; i < click_data.pts.size(); ++i) {
					valid_selections[i] = buildRoiFromClick(click_data.pts[i], static_cast<int>(i),
						preview_rects[i], original_rects[i]) ? 1 : 0;
				}
				processed_revision = click_data.revision;
			}

			cv::Mat display_img = small_vis.clone();
			for (size_t i = 0; i < click_data.pts.size(); ++i) {
				cv::circle(display_img, click_data.pts[i], 6, cv::Scalar(0, 255, 255), -1);
				cv::putText(display_img, std::to_string(i + 1), click_data.pts[i] + cv::Point(10, -10),
					cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

				if (valid_selections[i]) {
					cv::RotatedRect scaled_rect(
						preview_rects[i].center * static_cast<float>(scale),
						preview_rects[i].size * static_cast<float>(scale),
						preview_rects[i].angle);
					cv::Point2f vertices[4];
					scaled_rect.points(vertices);
					for (int j = 0; j < 4; ++j) {
						cv::line(display_img, vertices[j], vertices[(j + 1) % 4],
							cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
					}
				}
			}

			cv::imshow(winName, display_img);
			int key = cv::waitKey(20);
			if (key == 27) break;

			const bool all_selected = click_data.pts.size() == static_cast<size_t>(keep_num) &&
				std::all_of(valid_selections.begin(), valid_selections.end(), [](uchar valid) { return valid != 0; });
			if ((key == 13 || key == 10) && all_selected) {
				found_rois = original_rects;
				confirmed = true;
				break;
			}
		}
		cv::destroyWindow(winName);

		if (!confirmed) found_rois.clear();

		return found_rois;
	}

	std::cout << ">>> Global threshold fallback mode." << std::endl;
	cv::Mat combined_mask = makeFlatMask(grad_thresh, 5);

	cv::Mat labels, stats, centroids;
	int num_labels = cv::connectedComponentsWithStats(combined_mask, labels, stats, centroids, 8, CV_32S);

	struct Block { int id; double cx; double area; cv::RotatedRect rbox; };
	std::vector<Block> candidates;

	for (int i = 1; i < num_labels; i++) {
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		if (area <= min_area_thresh) {
			continue;
		}

		cv::Mat comp_mask = (labels == i);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(comp_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		if (contours.empty()) {
			continue;
		}

		int max_idx = 0;
		double max_area = 0.0;
		for (size_t c = 0; c < contours.size(); ++c) {
			double contour_area = cv::contourArea(contours[c]);
			if (contour_area > max_area) {
				max_area = contour_area;
				max_idx = static_cast<int>(c);
			}
		}

		cv::RotatedRect rbox = cv::minAreaRect(contours[max_idx]);
		float w = std::max(rbox.size.width, 1.0f);
		float h = std::max(rbox.size.height, 1.0f);
		float fill_ratio = static_cast<float>(max_area / (w * h));
		float aspect = std::max(w, h) / std::max(1.0f, std::min(w, h));

		if (fill_ratio < 0.35f || aspect > 12.0f) {
			continue;
		}

		candidates.push_back({ i, centroids.at<double>(i, 0), max_area, rbox });
	}

	std::sort(candidates.begin(), candidates.end(), [](const Block& a, const Block& b) {
		return a.cx < b.cx;
	});

	int actual_keep = std::min(keep_num, static_cast<int>(candidates.size()));
	for (int i = 0; i < actual_keep; i++) {
		found_rois.push_back(mapRectToOriginal(candidates[i].rbox));
	}

	return found_rois;
}
