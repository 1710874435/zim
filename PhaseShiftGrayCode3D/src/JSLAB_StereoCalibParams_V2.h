#pragma once
#include <opencv2/opencv.hpp>
#include "dllexport.h"

/**
  * @brief V2 独立标定参数集：包含基础几何参数与逐像素误差补偿图
  */
struct DLL_EXPORT JSLAB_StereoCalibParams_V2
{
	// V1 基础几何参数：左右相机内参、畸变、左右相机相对旋转与平移。
	cv::Mat camera_matrix[2];
		cv::Mat dist_coeffs[2];
		cv::Mat R;
		cv::Mat t;
		cv::Size image_size;

	// V2 逐像素三维坐标拟合系数图，类型为 CV_32FC4。
	// xyz_calib_coeffs[0]/[1]/[2] 分别对应 X、Y、Z 方程的 (c0,c1,c2,c3)。
	// 单个像素以视差 d 计算坐标：c0 + c1*d + c2*d*d + c3*d*d*d。

	cv::Mat xyz_calib_coeffs[3];
	// 从二进制文件读取矩阵类型、尺寸和连续像素数据。
	static bool loadMatBinary(const std::string& filename, cv::Mat& mat);
};
/**
  * @brief Load stereo calibration parameters from a file (OpenCV FileStorage XML/YAML)
  * @param [in]  filename Path to the calibration file
  * @param [out] params   Loaded calibration parameters
  * @return true on success, false if the file could not be opened
  */
bool DLL_EXPORT loadStereoCalibParams_V2(const std::string& filename, JSLAB_StereoCalibParams_V2& params);