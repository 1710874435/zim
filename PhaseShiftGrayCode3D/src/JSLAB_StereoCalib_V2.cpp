#include "JSLAB_StereoCalibParams_V1.h"
#include "JSLAB_StereoCalibParams_V2.h"
#include "JSLAB_StereoCalib_V2.h"
#include <iostream>
#include <fstream>

JSLAB_StereoCalib_V2::JSLAB_StereoCalib_V2(const cv::Size img_size) : 
	m_img_size(img_size) 
{

}

JSLAB_StereoCalib_V2::~JSLAB_StereoCalib_V2() {}

// 基础点位设置
void JSLAB_StereoCalib_V2::setTargetObjectPoints(const std::vector<cv::Point3f>& object_points)
{ 
	m_object_points = object_points;
}

// 内部辅助函数
void saveMatBinary(const std::string& filename, const cv::Mat& mat) {
	std::ofstream fs(filename, std::ios::binary);
	int type = mat.type(), rows = mat.rows, cols = mat.cols;
	fs.write((char*)&type, sizeof(int));
	fs.write((char*)&rows, sizeof(int));
	fs.write((char*)&cols, sizeof(int));
	fs.write((char*)mat.data, mat.step[0] * mat.rows);
	fs.close();
}

void JSLAB_StereoCalib_V2::addPlaneReferenceData(const cv::Mat& disp_map, const cv::Mat& mask)
{
	cv::Mat disp_32f;
	disp_map.convertTo(disp_32f, CV_32FC1);
	m_ref_disps.push_back(disp_32f);
	m_ref_masks.push_back(mask.clone());
}

void JSLAB_StereoCalib_V2::clearPlaneReferenceData()
{
	m_ref_disps.clear();
	m_ref_masks.clear();
}



void JSLAB_StereoCalib_V2::setTargetImagePoints(const std::vector<std::vector<cv::Point2f>>& left_image_points,
	const std::vector<std::vector<cv::Point2f>>& right_image_points)
{
	m_left_image_points = left_image_points;
	m_right_image_points = right_image_points;
}


void JSLAB_StereoCalib_V2::startBaseCalibration(double& lrerr, double& lerr, double& rerr)
{
	std::vector<std::vector<cv::Point3f>> object_points(m_left_image_points.size(), m_object_points);

	// 左相机单目标定：求内参、畸变参数和各拍摄组标靶姿态。
	std::vector<cv::Mat> left_rvecs;
	std::vector<cv::Mat> left_tvecs;
	lerr = cv::calibrateCamera(object_points, m_left_image_points, m_img_size,
		m_left_camera_matrix, m_left_dist_coeffs,
		left_rvecs, left_tvecs, 0,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, DBL_EPSILON));

	// 右相机执行同样的单目标定。
	std::vector<cv::Mat> right_rvecs;
	std::vector<cv::Mat> right_tvecs;
	rerr = cv::calibrateCamera(object_points, m_right_image_points, m_img_size,
		m_right_camera_matrix, m_right_dist_coeffs,
		right_rvecs, right_tvecs, 0,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, DBL_EPSILON));

	// 固定左右内参，联合左右对应点求相对旋转与平移。
	cv::Mat E, F;
	lrerr = cv::stereoCalibrate(object_points,
		m_left_image_points, m_right_image_points,
		m_left_camera_matrix, m_left_dist_coeffs,
		m_right_camera_matrix, m_right_dist_coeffs,
		m_img_size, m_R, m_T, E, F,
		cv::CALIB_USE_INTRINSIC_GUESS);

	std::cout << "left_err:" << lerr << std::endl;
	std::cout << "right_err:" << rerr << std::endl;
	std::cout << "stereo_err:" << lrerr << std::endl;
}

bool JSLAB_StereoCalib_V2::saveCalibResult(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened()) return false;

	fs << "LeftCameraMatrix" << m_left_camera_matrix;
	fs << "LeftDistortionCoefficients" << m_left_dist_coeffs;
	fs << "RightCameraMatrix" << m_right_camera_matrix;
	fs << "RightDistortionCoefficients" << m_right_dist_coeffs;
	fs << "RotationMatrix" << m_R;
	fs << "TranslationVector" << m_T;
	fs << "ImageSize" << m_img_size;
	fs.release();

	// 保存大矩阵为二进制
	std::string base_path = filename.substr(0, filename.find_last_of('.'));
	saveMatBinary(base_path + "_rx.bin", m_xyz_calib_coeffs[0]);
	saveMatBinary(base_path + "_ry.bin", m_xyz_calib_coeffs[1]);
	saveMatBinary(base_path + "_rz.bin", m_xyz_calib_coeffs[2]);

	return true;
}

void JSLAB_StereoCalib_V2::computePixelwiseDepthCalibration()
{
	std::cout << ">>> 正在执行 V2 逐像素深度标定 (极线校正系 + 法向投影 + d的三次方程)..." << std::endl;

	cv::Mat R1, R2, P1, P2, Q;
	cv::stereoRectify(m_left_camera_matrix, m_left_dist_coeffs, m_right_camera_matrix, m_right_dist_coeffs,
		m_img_size, m_R, m_T, R1, R2, P1, P2, Q, 0, -1.0, m_img_size);

	int num_planes = m_ref_disps.size();
	std::vector<cv::Vec4d> plane_eqs(num_planes);

	// 1. 拟合平面方程 (严格在极线校正后的坐标系下进行)
	for (int i = 0; i < num_planes; i++)
	{
		std::vector<cv::Point3d> pts_3d;
		pts_3d.reserve(m_img_size.width * m_img_size.height / 10);

		for (int r = 0; r < m_img_size.height; r++) {
			for (int c = 0; c < m_img_size.width; c++) {
				if (m_ref_masks[i].at<uchar>(r, c) == 0) continue;
				float d = m_ref_disps[i].at<float>(r, c);
				if (d <= 0) continue;

				// 仅通过 Q 矩阵计算校正坐标系下的 (X, Y, Z)，不乘 R1
				double w = Q.at<double>(3, 2) * d + Q.at<double>(3, 3);
				double x = (c + Q.at<double>(0, 3)) / w;
				double y = (r + Q.at<double>(1, 3)) / w;
				double z = Q.at<double>(2, 3) / w;

				pts_3d.push_back(cv::Point3d(x, y, z));
			}
		}

		cv::Mat points_mat(pts_3d.size(), 3, CV_64F);
		for (size_t k = 0; k < pts_3d.size(); ++k) {
			points_mat.at<double>(k, 0) = pts_3d[k].x;
			points_mat.at<double>(k, 1) = pts_3d[k].y;
			points_mat.at<double>(k, 2) = pts_3d[k].z;
		}

		// PCA 拟合平面
		cv::PCA pca(points_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);
		double A = pca.eigenvectors.at<double>(2, 0);
		double B = pca.eigenvectors.at<double>(2, 1);
		double C = pca.eigenvectors.at<double>(2, 2);
		double D = -(A * pca.mean.at<double>(0, 0) + B * pca.mean.at<double>(0, 1) + C * pca.mean.at<double>(0, 2));

		// PCA 的特征向量已经归一化，即 A^2 + B^2 + C^2 = 1
		plane_eqs[i] = cv::Vec4d(A, B, C, D);
	}

	// 初始化系数图矩阵 (注意：必须是 CV_32FC4，存放 4 个系数)
	m_xyz_calib_coeffs[0] = cv::Mat::zeros(m_img_size, CV_32FC4); // X_rect
	m_xyz_calib_coeffs[1] = cv::Mat::zeros(m_img_size, CV_32FC4); // Y_rect
	m_xyz_calib_coeffs[2] = cv::Mat::zeros(m_img_size, CV_32FC4); // Z_rect

	std::cout << ">>> 开启 OpenMP 多核并发拟合 (2600万像素)..." << std::endl;

#pragma omp parallel for collapse(2)
	for (int r = 0; r < m_img_size.height; r++) {
		for (int c = 0; c < m_img_size.width; c++) {

			std::vector<double> X_d, Y_x, Y_y, Y_z;
			X_d.reserve(num_planes);
			Y_x.reserve(num_planes);
			Y_y.reserve(num_planes);
			Y_z.reserve(num_planes);

			for (int i = 0; i < num_planes; i++) {
				if (m_ref_masks[i].at<uchar>(r, c) == 0) continue;
				double d_raw = m_ref_disps[i].at<float>(r, c);
				if (d_raw <= 0) continue;

				// 原始校正系坐标
				double w = Q.at<double>(3, 2) * d_raw + Q.at<double>(3, 3);
				double x_raw = (c + Q.at<double>(0, 3)) / w;
				double y_raw = (r + Q.at<double>(1, 3)) / w;
				double z_raw = Q.at<double>(2, 3) / w;

				double A = plane_eqs[i][0], B = plane_eqs[i][1], C = plane_eqs[i][2], D = plane_eqs[i][3];

				// 沿平面法向量投影，得到理想平面上的对应三维坐标样本。
				// 因为 A^2+B^2+C^2=1，可以直接省略除法
				double dist = A * x_raw + B * y_raw + C * z_raw + D;

				double x_ideal = x_raw - A * dist;
				double y_ideal = y_raw - B * dist;
				double z_ideal = z_raw - C * dist;

				X_d.push_back(d_raw); // 自变量直接用 d
				Y_x.push_back(x_ideal);
				Y_y.push_back(y_ideal);
				Y_z.push_back(z_ideal);
			}

			// 需要至少 4 个点才能拟合三次多项式
			if (X_d.size() >= 4) {
				int n_pts = X_d.size();

				cv::Mat A_mat(n_pts, 4, CV_64F); // 4 列
				cv::Mat B_mat_x(n_pts, 1, CV_64F), B_mat_y(n_pts, 1, CV_64F), B_mat_z(n_pts, 1, CV_64F);

				for (int k = 0; k < n_pts; k++) {
					double d_val = X_d[k];
					A_mat.at<double>(k, 0) = 1.0;
					A_mat.at<double>(k, 1) = d_val;
					A_mat.at<double>(k, 2) = d_val * d_val;
					A_mat.at<double>(k, 3) = d_val * d_val * d_val; // 三次项

					B_mat_x.at<double>(k, 0) = Y_x[k];
					B_mat_y.at<double>(k, 0) = Y_y[k];
					B_mat_z.at<double>(k, 0) = Y_z[k];
				}

				cv::Mat c_x, c_y, c_z;
				cv::solve(A_mat, B_mat_x, c_x, cv::DECOMP_SVD);
				cv::solve(A_mat, B_mat_y, c_y, cv::DECOMP_SVD);
				cv::solve(A_mat, B_mat_z, c_z, cv::DECOMP_SVD);

				// 写入 CV_32FC4 矩阵
				m_xyz_calib_coeffs[0].at<cv::Vec4f>(r, c) = cv::Vec4f((float)c_x.at<double>(0, 0), (float)c_x.at<double>(1, 0), (float)c_x.at<double>(2, 0), (float)c_x.at<double>(3, 0));
				m_xyz_calib_coeffs[1].at<cv::Vec4f>(r, c) = cv::Vec4f((float)c_y.at<double>(0, 0), (float)c_y.at<double>(1, 0), (float)c_y.at<double>(2, 0), (float)c_y.at<double>(3, 0));
				m_xyz_calib_coeffs[2].at<cv::Vec4f>(r, c) = cv::Vec4f((float)c_z.at<double>(0, 0), (float)c_z.at<double>(1, 0), (float)c_z.at<double>(2, 0), (float)c_z.at<double>(3, 0));
			}
		}
	}
	std::cout << ">>> 并发计算完成！逐像素 X/Y/Z 三次方程拟合完毕！" << std::endl;
}
// 复制内部参数，避免调用者持有与标定器共享的矩阵数据。
JSLAB_StereoCalibParams_V2 JSLAB_StereoCalib_V2::getParams() const
{
	JSLAB_StereoCalibParams_V2 params;

	m_left_camera_matrix.copyTo(params.camera_matrix[0]);
	m_right_camera_matrix.copyTo(params.camera_matrix[1]);

	m_left_dist_coeffs.copyTo(params.dist_coeffs[0]);
	m_right_dist_coeffs.copyTo(params.dist_coeffs[1]);

	m_R.copyTo(params.R);
	m_T.copyTo(params.t);

	params.image_size = m_img_size;

	if (!m_xyz_calib_coeffs[0].empty()) {
		m_xyz_calib_coeffs[0].copyTo(params.xyz_calib_coeffs[0]);
		m_xyz_calib_coeffs[1].copyTo(params.xyz_calib_coeffs[1]);
		m_xyz_calib_coeffs[2].copyTo(params.xyz_calib_coeffs[2]);
	}
	return params;
}

// =======================================================
// V2 使用与 V1 相同的几何参数诊断图表。
// =======================================================
void JSLAB_StereoCalib_V2::visualizeCalibration(bool show_windows)
{
	if (!show_windows)
		return;

	if (m_left_image_points.empty() || m_left_camera_matrix.empty())
	{
		std::cout << "[Warning] No calibration data available for visualization." << std::endl;
		return;
	}

	std::cout << "------------------------------------------------" << std::endl;
	std::cout << "Generating calibration visualization (Extrinsics & Reprojection Errors)..." << std::endl;

	// =========================================================
	// 图 1：左右相机相对位置的俯视图。
	// =========================================================
	int H = 600, W = 600;
	cv::Mat canvas_ext = cv::Mat::zeros(H, W, CV_8UC3);

	cv::Point2f center(W / 2, H - 100);
	double tx = m_T.at<double>(0, 0);
	double ty = m_T.at<double>(1, 0);
	double tz = m_T.at<double>(2, 0);
	double baseline = sqrt(tx * tx + ty * ty + tz * tz);
	double scale = (W / 2.5) / (baseline + 1e-5);

	// 左相机作为参考原点，以红色绘制。
	cv::circle(canvas_ext, center, 8, cv::Scalar(0, 0, 255), -1);
	cv::putText(canvas_ext, "Left Cam (Ref)", center + cv::Point2f(-40, 25),
		cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
	cv::line(canvas_ext, center, center + cv::Point2f(-60, -120), cv::Scalar(0, 0, 150), 1);
	cv::line(canvas_ext, center, center + cv::Point2f(60, -120), cv::Scalar(0, 0, 150), 1);
	cv::putText(canvas_ext, "Z Axis (Depth)", center + cv::Point2f(70, -100),
		cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

	// 根据相对平移向量计算右相机位置，以蓝色绘制。
	cv::Point2f right_cam_pos;
	right_cam_pos.x = center.x + tx * scale;
	right_cam_pos.y = center.y - tz * scale;

	cv::circle(canvas_ext, right_cam_pos, 8, cv::Scalar(255, 0, 0), -1);
	cv::putText(canvas_ext, "Right Cam", right_cam_pos + cv::Point2f(-30, 25),
		cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 1);
	cv::line(canvas_ext, right_cam_pos, right_cam_pos + cv::Point2f(-60, -120), cv::Scalar(150, 0, 0), 1);
	cv::line(canvas_ext, right_cam_pos, right_cam_pos + cv::Point2f(60, -120), cv::Scalar(150, 0, 0), 1);

	// 连接两个相机中心，并显示平移向量与基线长度。
	cv::line(canvas_ext, center, right_cam_pos, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

	std::string t_str = "T = [" + std::to_string(tx).substr(0, 6) + ", "
		+ std::to_string(ty).substr(0, 4) + ", "
		+ std::to_string(tz).substr(0, 6) + "]";
	std::string b_str = "Baseline: " + std::to_string(baseline).substr(0, 6) + " mm";

	cv::putText(canvas_ext, "System Calibration Result (Top-Down)",
		cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
	cv::putText(canvas_ext, t_str, cv::Point(20, 60),
		cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
	cv::putText(canvas_ext, b_str, cv::Point(20, 80),
		cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

	// =========================================================
	// 图 2：左相机每张标定图的重投影误差柱状图。
	// =========================================================
	cv::Mat canvas_err = cv::Mat::zeros(400, 800, CV_8UC3);
	int num_imgs = (int)m_left_image_points.size();
	double max_err = 0;
	double total_avg = 0;
	std::vector<double> errors_per_view;

	for (int i = 0; i < num_imgs; i++)
	{
		std::vector<cv::Point2f> proj_pts;
		cv::Mat rvec, tvec;
		cv::solvePnP(m_object_points, m_left_image_points[i],
			m_left_camera_matrix, m_left_dist_coeffs, rvec, tvec);
		cv::projectPoints(m_object_points, rvec, tvec,
			m_left_camera_matrix, m_left_dist_coeffs, proj_pts);

		double err = cv::norm(m_left_image_points[i], proj_pts, cv::NORM_L2) / sqrt(proj_pts.size());
		errors_per_view.push_back(err);
		total_avg += err;
		if (err > max_err) max_err = err;
	}
	if (num_imgs > 0) total_avg /= num_imgs;

	int margin_left = 60;
	int margin_bottom = 50;
	int plot_w = canvas_err.cols - margin_left - 20;
	int plot_h = canvas_err.rows - margin_bottom - 50;
	int bar_width = (plot_w / (num_imgs > 0 ? num_imgs : 1)) - 5;
	if (bar_width < 2) bar_width = 2;

	double y_max_val = max_err * 1.2;
	if (y_max_val < 0.1) y_max_val = 0.5;

	for (int i = 0; i < num_imgs; i++)
	{
		double err = errors_per_view[i];
		int bar_height = (int)((err / y_max_val) * plot_h);
		int x1 = margin_left + i * (bar_width + 5) + 5;
		int y1 = canvas_err.rows - margin_bottom - bar_height;
		cv::Scalar color = (err > 0.1) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

		cv::rectangle(canvas_err, cv::Rect(x1, y1, bar_width, bar_height), color, -1);

		if (num_imgs < 20 || i % 2 == 0)
		{
			cv::putText(canvas_err, std::to_string(i + 1),
				cv::Point(x1, canvas_err.rows - margin_bottom + 15),
				cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
		}
	}

	// 绘制误差柱状图的横轴和纵轴。
	cv::line(canvas_err,
		cv::Point(margin_left, canvas_err.rows - margin_bottom),
		cv::Point(canvas_err.cols - 10, canvas_err.rows - margin_bottom),
		cv::Scalar(255, 255, 255), 1);
	cv::line(canvas_err,
		cv::Point(margin_left, canvas_err.rows - margin_bottom),
		cv::Point(margin_left, 40),
		cv::Scalar(255, 255, 255), 1);

	cv::putText(canvas_err, "Reprojection Errors per Image (Left Camera)",
		cv::Point(margin_left, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

	std::string stat_str = "Mean: " + std::to_string(total_avg).substr(0, 5)
		+ " px  |  Max: " + std::to_string(max_err).substr(0, 5) + " px";
	cv::putText(canvas_err, stat_str, cv::Point(margin_left, canvas_err.rows - 10),
		cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

	// =========================================================
	// 显示诊断窗口，并等待用户按任意键关闭。
	// =========================================================
	cv::namedWindow("Figure 1: Extrinsics", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Figure 2: Reprojection Errors", cv::WINDOW_AUTOSIZE);
	cv::imshow("Figure 1: Extrinsics", canvas_ext);
	cv::imshow("Figure 2: Reprojection Errors", canvas_err);
	cv::moveWindow("Figure 1: Extrinsics", 100, 100);
	cv::moveWindow("Figure 2: Reprojection Errors", 710, 100);

	std::cout << ">> Visualization ready. Press any key to continue..." << std::endl;
	cv::waitKey(0);

	cv::destroyWindow("Figure 1: Extrinsics");
	cv::destroyWindow("Figure 2: Reprojection Errors");
	std::cout << "------------------------------------------------" << std::endl;
}

bool JSLAB_StereoCalib_V2::loadBaseCalibration(const std::string& filename)
{
	JSLAB_StereoCalibParams_V1 params_v1;
	if (!loadStereoCalibParams_V1(filename, params_v1)) {
		return false;
	}

	// 将读取到的 V1 参数直接注入给 V2 的私有成员
	params_v1.camera_matrix[0].copyTo(m_left_camera_matrix);
	params_v1.camera_matrix[1].copyTo(m_right_camera_matrix);
	params_v1.dist_coeffs[0].copyTo(m_left_dist_coeffs);
	params_v1.dist_coeffs[1].copyTo(m_right_dist_coeffs);
	params_v1.R.copyTo(m_R);
	params_v1.t.copyTo(m_T);
	m_img_size = params_v1.image_size;

	return true;
}

