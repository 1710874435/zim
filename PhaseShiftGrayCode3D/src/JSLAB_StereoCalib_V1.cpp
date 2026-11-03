#include "JSLAB_StereoCalib_V1.h"
#include <iostream>

JSLAB_StereoCalib_V1::JSLAB_StereoCalib_V1(const cv::Size img_size) :
	m_img_size(img_size)
{

}

void JSLAB_StereoCalib_V1::setTargetObjectPoints(const std::vector<cv::Point3f>& object_points)
{
	// 保存一块标靶自身坐标系中的点位，所有拍摄组复用这份几何尺寸。
	m_object_points = object_points;
}

void JSLAB_StereoCalib_V1::setTargetImagePoints(const std::vector<std::vector<cv::Point2f>>& left_image_points,
                                                 const std::vector<std::vector<cv::Point2f>>& right_image_points)
{
	// 保存每个标靶点在每组左右图中的对应像素位置。
	m_left_image_points  = left_image_points;
	m_right_image_points = right_image_points;
}

void JSLAB_StereoCalib_V1::startCalib(double& lrerr, double& lerr, double& rerr)
{
	// 为每一组二维观测复制同一份标靶三维点坐标。
	std::vector<std::vector<cv::Point3f>> object_points(m_left_image_points.size(), m_object_points);

	// 左相机：求内参、畸变，以及每组标靶相对于左相机的姿态。
	std::vector<cv::Mat> left_rvecs;
	std::vector<cv::Mat> left_tvecs;
	lerr = cv::calibrateCamera(object_points, m_left_image_points, m_img_size,
	                           m_left_camera_matrix, m_left_dist_coeffs,
	                           left_rvecs, left_tvecs, 0,
	                           cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, DBL_EPSILON));

	// 右相机执行同样的单目标定流程。
	std::vector<cv::Mat> right_rvecs;
	std::vector<cv::Mat> right_tvecs;
	rerr = cv::calibrateCamera(object_points, m_right_image_points, m_img_size,
	                           m_right_camera_matrix, m_right_dist_coeffs,
	                           right_rvecs, right_tvecs, 0,
	                           cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, DBL_EPSILON));

	// 固定左右内参，联合左右对应点求左右相机的相对旋转 m_R 与平移 m_T。
	cv::Mat E, F;
	lrerr = cv::stereoCalibrate(object_points,
	                            m_left_image_points, m_right_image_points,
	                            m_left_camera_matrix, m_left_dist_coeffs,
	                            m_right_camera_matrix, m_right_dist_coeffs,
	                            m_img_size, m_R, m_T, E, F,
	                            cv::CALIB_USE_INTRINSIC_GUESS);

	std::cout << "left_err:"  << lerr  << std::endl;
	std::cout << "right_err:" << rerr  << std::endl;
	std::cout << "stereo_err:" << lrerr << std::endl;
}

bool JSLAB_StereoCalib_V1::saveCalibResult(const std::string& filename)
{
	// 写出重建所需的全部 V1 几何参数。
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened())
		return false;

	fs << "LeftCameraMatrix"           << m_left_camera_matrix;
	fs << "LeftDistortionCoefficients" << m_left_dist_coeffs;
	fs << "RightCameraMatrix"          << m_right_camera_matrix;
	fs << "RightDistortionCoefficients"<< m_right_dist_coeffs;
	fs << "RotationMatrix"             << m_R;
	fs << "TranslationVector"          << m_T;
	fs << "ImageSize"                  << m_img_size;

	fs.release();
	return true;
}

void JSLAB_StereoCalib_V1::visualizeCalibration(bool show_windows)
{
	// show_windows 控制是否创建 OpenCV 可视化窗口。
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
	// 图 1：以左相机为参考原点，绘制左右相机的俯视相对位置。
	// =========================================================
	int H = 600, W = 600;
	cv::Mat canvas_ext = cv::Mat::zeros(H, W, CV_8UC3);

	cv::Point2f center(W / 2, H - 100);
	double tx       = m_T.at<double>(0, 0);
	double ty       = m_T.at<double>(1, 0);
	double tz       = m_T.at<double>(2, 0);
	double baseline = sqrt(tx * tx + ty * ty + tz * tz);
	double scale    = (W / 2.5) / (baseline + 1e-5);

	// Left camera (red) world origin
	cv::circle(canvas_ext, center, 8, cv::Scalar(0, 0, 255), -1);
	cv::putText(canvas_ext, "Left Cam (Ref)", center + cv::Point2f(-40, 25),
	            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
	cv::line(canvas_ext, center, center + cv::Point2f(-60, -120), cv::Scalar(0, 0, 150), 1);
	cv::line(canvas_ext, center, center + cv::Point2f( 60, -120), cv::Scalar(0, 0, 150), 1);
	cv::putText(canvas_ext, "Z Axis (Depth)", center + cv::Point2f(70, -100),
	            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(100, 100, 100), 1);

	// Right camera (blue) position from R and T
	cv::Point2f right_cam_pos;
	right_cam_pos.x = center.x + tx * scale;
	right_cam_pos.y = center.y - tz * scale;

	cv::circle(canvas_ext, right_cam_pos, 8, cv::Scalar(255, 0, 0), -1);
	cv::putText(canvas_ext, "Right Cam", right_cam_pos + cv::Point2f(-30, 25),
	            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 1);
	cv::line(canvas_ext, right_cam_pos, right_cam_pos + cv::Point2f(-60, -120), cv::Scalar(150, 0, 0), 1);
	cv::line(canvas_ext, right_cam_pos, right_cam_pos + cv::Point2f( 60, -120), cv::Scalar(150, 0, 0), 1);

	// Baseline line and text info
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
	// 图 2：逐组重投影误差柱状图，用于定位质量较差的标定图。
	// =========================================================
	cv::Mat canvas_err = cv::Mat::zeros(400, 800, CV_8UC3);
	int num_imgs       = (int)m_left_image_points.size();
	double max_err     = 0;
	double total_avg   = 0;
	std::vector<double> errors_per_view;//创建宽 800、高 400 的黑色三通道 BGR 画布，num_imgs 是标定图片组数，errors_per_view 用来保存每组误差，max_err 与 total_avg 用来统计最大值和均方根误差的平均值。

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

	int margin_left   = 60;
	int margin_bottom = 50;
	int plot_w        = canvas_err.cols - margin_left - 20;
	int plot_h        = canvas_err.rows - margin_bottom - 50;
	int bar_width     = (plot_w / (num_imgs > 0 ? num_imgs : 1)) - 5;
	if (bar_width < 2) bar_width = 2;

	double y_max_val = max_err * 1.2;
	if (y_max_val < 0.1) y_max_val = 0.5;

	for (int i = 0; i < num_imgs; i++)
	{
		double err       = errors_per_view[i];
		int bar_height   = (int)((err / y_max_val) * plot_h);
		int x1           = margin_left + i * (bar_width + 5) + 5;
		int y1           = canvas_err.rows - margin_bottom - bar_height;
		cv::Scalar color = (err > 0.1) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

		cv::rectangle(canvas_err, cv::Rect(x1, y1, bar_width, bar_height), color, -1);

		if (num_imgs < 20 || i % 2 == 0)
		{
			cv::putText(canvas_err, std::to_string(i + 1),
			            cv::Point(x1, canvas_err.rows - margin_bottom + 15),
			            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
		}
	}

	// Axes
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
	// 显示两张诊断图，并等待用户按键关闭。
	// =========================================================
	cv::namedWindow("Figure 1: Extrinsics",          cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Figure 2: Reprojection Errors", cv::WINDOW_AUTOSIZE);
	cv::imshow("Figure 1: Extrinsics",          canvas_ext);
	cv::imshow("Figure 2: Reprojection Errors", canvas_err);
	cv::moveWindow("Figure 1: Extrinsics",          100, 100);
	cv::moveWindow("Figure 2: Reprojection Errors", 710, 100);

	std::cout << ">> Visualization ready. Press any key to continue..." << std::endl;
	cv::waitKey(0);

	cv::destroyWindow("Figure 1: Extrinsics");
	cv::destroyWindow("Figure 2: Reprojection Errors");
	std::cout << "------------------------------------------------" << std::endl;
}

JSLAB_StereoCalib_V1::~JSLAB_StereoCalib_V1()
{

}
