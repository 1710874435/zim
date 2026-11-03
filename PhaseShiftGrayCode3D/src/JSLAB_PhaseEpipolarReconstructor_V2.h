#pragma once

#include <opencv.hpp>
#include "JSLAB_StereoCalibParams_V1.h"
#include "JSLAB_StereoCalibParams_V2.h"
#include "dllexport.h"

/**
  * @brief V2 phase epipolar reconstructor (order-map based stereo matching)
  */
class DLL_EXPORT JSLAB_PhaseEpipolarReconstructor_V2
{
public:

	/**
	  * @brief Configuration parameters
	  */
	struct Config
	{
		// Depth range clamp (camera-coordinate Z, same unit as calibration)
		double min_depth = 0;
		double max_depth = 1000;

		// Adjacent-pixel phase difference limits used to filter unreliable pixels
		double min_phase_diff = 0.03;
		double max_phase_diff = 0.4;

		// Maximum allowed phase residual between matched left and right pixels
		double grad_threshold = 1;//�Ӳ�ƥ�����λ�����ֵ

		// When true:  output x/y/z (rectified coordinate system) and filter on z.
		// When false: output rx/ry/rz (original camera coordinate system, R1^T applied)
		//             and filter on rz.  (default, preserves original behaviour)
		bool use_rectified_coords = true;

		// When true: apply a 5*5 median blur to the disparity map before reprojection
		//            to suppress phase-matching noise at the cost of slight blurring.
		bool use_disp_median_filter = true;

		bool use_full_left_rectification = true;

		// Radius outlier filter parameters
		float radius_search      = 0.7f;   // 3-D search radius (same unit as calibration, e.g. mm)
		int   radius_min_neighbors = 10;     // min neighbours to keep a point
		int   radius_window_size = 10;       // pixel window half-size for neighbour search

		// MLS smoothing (applied after radius outlier filter)
		bool  use_mls            = false;
		int   mls_window_size    = 30;       // half-size of pixel search window
		float mls_spatial_sigma  = 0.6f;   // Gaussian weight sigma in 3-D space
		bool  use_gpu_mls        = true;   // true = CUDA, false = CPU (OpenMP)
		int   mls_polynomial_order = 1;     // 0 = plane, 1 = bilinear, 2 = biquadratic
		float mls_max_radius     = 0.0f;    // hard 3-D cutoff; 0 = auto (3*sigma)
		int   mls_iterations     = 3;       // number of MLS passes (2-3 recommended)
	};

	JSLAB_PhaseEpipolarReconstructor_V2(const Config& config);

	~JSLAB_PhaseEpipolarReconstructor_V2();

	/**
	  * @brief Set calibration parameters (triggers rectification map computation)
	  * @param params Stereo calibration parameters
	  */
	void setCalibrationParams(const JSLAB_StereoCalibParams_V1& params);

	/**
	  * @brief 导入 V2 极速高精度标定参数 (包含逐像�?XYZ 多项�?
	  */
	void setCalibrationParamsV2(const JSLAB_StereoCalibParams_V2& params);

	/**
	  * @brief Set phase maps, order maps and modulation map
	  * @param left     Left camera absolute phase map (CV_64FC1)
	  * @param right    Right camera absolute phase map (CV_64FC1)
	  * @param left_k   Left camera fringe order map (CV_32SC1)
	  * @param right_k  Right camera fringe order map (CV_32SC1)
	  * @param left_mod Left camera modulation map (CV_64FC1)
	  * @param pha_f    Fringe frequency used during decoding
	  */
	void setData(const cv::Mat& left, const cv::Mat& right,
	             const cv::Mat& left_k, const cv::Mat& right_k,
	             const cv::Mat& left_mod, int pha_f);

	/**
	  * @brief Run 3-D reconstruction
	  * @param [out] xyz Output 3-D point cloud (CV_32FC3)
	  * @param [out] P   Valid pixel mask (CV_8UC1, non-zero = valid)
	  */
	void startCalculate(cv::Mat& xyz, cv::Mat& P);

	/**
	  * @brief V2 重建函数：实现多项式拟合解算 + 传统逻辑保底回退
	  */
	void startCalculate_v2(cv::Mat& xyz, cv::Mat& P);

	struct RectifiedROI
	{
		cv::Rect left_rect;
		cv::Rect right_rect;
		cv::Mat mask;
	};

	bool prepareRectifiedROI(const cv::RotatedRect& roi_original, RectifiedROI& roi) const;

	/**
	  * @brief V2 reconstruction restricted to one ROI.
	  */
	void startCalculate_v2_roi(cv::Mat& xyz, cv::Mat& P, const cv::RotatedRect& roi_original);
	void startCalculate_v2_roi(cv::Mat& xyz, cv::Mat& P, const RectifiedROI& roi);

	/**
	  * @brief 获取内部生成的视差图 (供外部做分割或检查使�?
	  */
	void getDisparityMap(cv::Mat& disp) const;
	void getRectifiedLeftModulation(cv::Mat& mod) const;
	void rectifyLeftImage(const cv::Mat& image, cv::Mat& rectified) const;

	/**
	  * @brief 自动检测视差图中的阶梯�?ROI
	  * @param keep_num        保留最左侧�?ROI 数量
	  * @param min_area_thresh 面积过滤阈�?
	  * @param grad_thresh     梯度阈值，用于切断阶梯间的粘连
	  * @return 返回识别到的目标区域矩形列表
	  */
	std::vector<cv::RotatedRect> detectTargetROIs(int keep_num = 2, int min_area_thresh = 50000, float grad_thresh = 1.0f);
private:

	/**
	  * @brief Estimate the valid disparity range from the configured depth bounds
	  */
	void estimateDisparityRange();

	/**
	  * @brief Order-constrained stereo matching with sub-pixel interpolation
	  * @param l_phase Left absolute phase map (CV_64FC1)
	  * @param r_phase Right absolute phase map (CV_64FC1)
	  * @param l_k     Left fringe order map (CV_32SC1)
	  * @param r_k     Right fringe order map (CV_32SC1)
	  * @param pha_f   Fringe frequency parameter
	  * @param disp    Output disparity map (CV_64FC1)
	  */
	void stereoMatch(const cv::Mat& l_phase, const cv::Mat& r_phase,
	                 const cv::Mat& l_k, const cv::Mat& r_k,
	                 int pha_f, cv::Mat& disp);


private:

	const Config m_config;
	JSLAB_StereoCalibParams_V1 m_params;
	JSLAB_StereoCalibParams_V2 m_params_v2; // 保存 V2 参数
	cv::Mat m_left_remap[2];
	cv::Mat m_right_remap[2];
	cv::Mat m_R1;
	cv::Mat m_Q;
	cv::Mat m_left;
	cv::Mat m_right;
	cv::Mat m_left_k;      // Rectified left fringe order map
	cv::Mat m_right_k;     // Rectified right fringe order map
	cv::Mat m_left_mod;    // Rectified left modulation map
	cv::Mat m_dispmap;
	int m_pha_f;           // Fringe frequency parameter
	cv::Mat m_xyz;
	cv::Mat m_P;
	cv::Mat m_region_mask;
	bool m_is_init;
	bool m_use_v2_coeffs = false;           // 标志位：当前是否处于 V2 高精度模�?
	double m_min_disparity;
	double m_max_disparity;
};
