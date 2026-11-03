#pragma once

#include <opencv.hpp>
#include "dllexport.h"

/**
  * @brief Stereo camera calibration parameter set
  */
struct DLL_EXPORT JSLAB_StereoCalibParams_V1
{
	// camera_matrix[0]/[1] 分别为左、右相机内参矩阵 K。
	cv::Mat camera_matrix[2];
	// dist_coeffs[0]/[1] 保存左右镜头的径向与切向畸变系数。
	cv::Mat dist_coeffs[2];
	// 左相机坐标系转换到右相机坐标系所需的旋转矩阵和平移向量。
	cv::Mat R;
	cv::Mat t;
	// 标定图像的像素宽度和高度，重建图像需要保持相同分辨率。
	cv::Size image_size;
};

/**
  * @brief Load stereo calibration parameters from a file (OpenCV FileStorage XML/YAML)
  * @param [in]  filename Path to the calibration file
  * @param [out] params   Loaded calibration parameters
  * @return true on success, false if the file could not be opened
  */
bool DLL_EXPORT loadStereoCalibParams_V1(const std::string& filename, JSLAB_StereoCalibParams_V1& params);
