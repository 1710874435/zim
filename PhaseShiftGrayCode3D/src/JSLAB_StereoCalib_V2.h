#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "JSLAB_StereoCalibParams_V2.h"
#include "dllexport.h"

class DLL_EXPORT JSLAB_StereoCalib_V2
{
public:
	// V2 先求解普通双目几何参数，再使用多个已知平面拟合逐像素坐标方程。
	JSLAB_StereoCalib_V2(const cv::Size img_size);
	~JSLAB_StereoCalib_V2();

	// --- 基础功能：同 V1 ---
	/**
	 * @brief 设置标靶坐标系中的三维点
	 * @param [in] object_points 标靶三维点集合，全部拍摄组复用
	 */
	void setTargetObjectPoints(const std::vector<cv::Point3f>& object_points);
	/**
		  * @brief 设置左右相机检测到的二维点
		  * @param [in] left_image_points  左图二维点，外层下标表示拍摄组
		  * @param [in] right_image_points 右图二维点，内层下标表示标靶点编号
		  */
	void setTargetImagePoints(const std::vector<std::vector<cv::Point2f>>& left_image_points,
							  const std::vector<std::vector<cv::Point2f>>& right_image_points);

	// =======================================================
	// --- 直接加载已有的基础几何标定结果 ---
	// =======================================================
	bool loadBaseCalibration(const std::string& filename);

	// --- V2 核心功能：混合标定 ---


	/**
	  * @brief 添加一个高度位置的参考平面数据 (需调用 N 次)
	  * @param disp_map 拍摄标准平直面得到的视差图 (CV_32F/64F)
	  * @param mask     有效区域掩膜 (CV_8U)
	  */
	// 保存一个参考平面的视差图与有效像素掩码，供后续系数拟合使用。
	void addPlaneReferenceData(const cv::Mat& disp_map, const cv::Mat& mask);
	void startBaseCalibration(double& lrerr, double& lerr, double& rerr);
	/**
	  * @brief 清空已加载的参考平面数据
	  */
	void clearPlaneReferenceData();
	// 对每个像素以多个平面的视差样本拟合 X/Y/Z 三次方程。
	void computePixelwiseDepthCalibration();


	/**
	  * @brief 保存标定结果到 XML 和 BIN 文件
	  * @param [in] filename 输出 XML 文件路径
	  * @return 参数和三张系数图保存成功时返回 true
	  */
	bool saveCalibResult(const std::string& filename);

	/**
	  * @brief 显示标定诊断结果：相机相对位置与逐图重投影误差
      *          - 图 1：左右相机位置俯视图
      *          - 图 2：左相机逐图重投影误差柱状图
      * @param [in] show_windows 为 false 时跳过窗口显示
      *                          。
      */
	void visualizeCalibration(bool show_windows = true);

	/**
	  * @brief 提取所有标定参数
	  */
	// 复制当前基础几何参数和逐像素系数，形成可交给重建器的参数对象。
	JSLAB_StereoCalibParams_V2 getParams() const;


private:

	cv::Size m_img_size;
	std::vector<cv::Point3f> m_object_points;
	std::vector<std::vector<cv::Point2f>> m_left_image_points;
	std::vector<std::vector<cv::Point2f>> m_right_image_points;

	cv::Mat m_left_camera_matrix;
	cv::Mat m_right_camera_matrix;
	cv::Mat m_left_dist_coeffs;
	cv::Mat m_right_dist_coeffs;
	cv::Mat m_R;
	cv::Mat m_T;

	// 外层下标对应参考平面编号；每个元素保留整张视差图或有效区域掩码。
	std::vector<cv::Mat> m_ref_disps;
	std::vector<cv::Mat> m_ref_masks;

	// V2 核心产物：X、Y、Z 的逐像素三次方程系数图，类型为 CV_32FC4。


	cv::Mat m_xyz_calib_coeffs[3];
};