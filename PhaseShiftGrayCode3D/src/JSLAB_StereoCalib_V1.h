#pragma once

#include <opencv.hpp>
#include "dllexport.h"

/**
  * @brief Stereo camera calibration (V1)
  */
class DLL_EXPORT JSLAB_StereoCalib_V1
{
public:

	/**
	  * @brief Constructor
	  * @param [in] img_size Image resolution
	  */
	JSLAB_StereoCalib_V1(const cv::Size img_size);

	/**
	  * @brief Set the 3-D world coordinates of the calibration target points
	  * @param [in] object_points 3-D point set of the calibration target
	  */
	void setTargetObjectPoints(const std::vector<cv::Point3f>& object_points);

	/**
	  * @brief Set the detected 2-D image points from left and right cameras
	  * @param [in] left_image_points  Detected points in the left camera images
	  * @param [in] right_image_points Detected points in the right camera images
	  */
	void setTargetImagePoints(const std::vector<std::vector<cv::Point2f>>& left_image_points,
	                          const std::vector<std::vector<cv::Point2f>>& right_image_points);

	/**
	  * @brief Run stereo calibration
	  * @param [out] lrerr Stereo reprojection error (both cameras)
	  * @param [out] lerr  Left camera reprojection error
	  * @param [out] rerr  Right camera reprojection error
	  */
	void startCalib(double& lrerr, double& lerr, double& rerr);

	/**
	  * @brief Save calibration results to a file (OpenCV FileStorage XML/YAML)
	  * @param [in] filename Output file path
	  * @return true on success, false if the file could not be opened
	  */
	bool saveCalibResult(const std::string& filename);

	/**
	  * @brief Visualize calibration results:
	  *          - Figure 1: top-down extrinsics view (left/right camera positions)
	  *          - Figure 2: per-image reprojection error bar chart (left camera)
	  * @param [in] show_windows If false the function returns immediately without
	  *                          opening any display window.
	  */
	void visualizeCalibration(bool show_windows = true);

	/**
	  * @brief Destructor
	  */
	~JSLAB_StereoCalib_V1();

private:
	// 输入图像分辨率与单张标靶的真实三维点坐标。
	cv::Size m_img_size;
	std::vector<cv::Point3f> m_object_points;
	// 外层下标表示拍摄序号，内层下标表示同一个标靶点的像素坐标。
	std::vector<std::vector<cv::Point2f>> m_left_image_points;
	std::vector<std::vector<cv::Point2f>> m_right_image_points;
	// V1 标定产物：左右相机内参、畸变参数和左右相机相对位姿。
	cv::Mat m_left_camera_matrix;
	cv::Mat m_right_camera_matrix;
	cv::Mat m_left_dist_coeffs;
	cv::Mat m_right_dist_coeffs;
	cv::Mat m_R;
	cv::Mat m_T;
};
