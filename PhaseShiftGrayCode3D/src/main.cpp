#include <iostream>
#include "JSLAB_CircleGridDetector.h"
#include "JSLAB_StereoCalib_V1.h"
#include "JSLAB_StereoCalib_V2.h"
#include "JSLAB_PhaseShiftDecoder.h"
#include "JSLAB_PhasePreview.h"
#include "JSLAB_PhaseEpipolarReconstructor_V1.h"
#include "JSLAB_PhaseEpipolarReconstructor_V2.h"
#include "JSLAB_StereoCalibParams_V1.h"
#include "JSLAB_StereoCalibParams_V2.h"
#include <fstream>
#include <opencv2/highgui.hpp>
#include <direct.h> // Windows下创建文件夹需要
#include <cstdio> // 引入 C 标准 I/O
#include <string>
#include <sys/stat.h>

// _mkdir 仅创建末级目录；父路径缺失时失败。逐级创建完整路径。
static void ensure_dir(const std::string& path)
{
	if (path.empty()) return;
	std::string cur;
	cur.reserve(path.size());
	for (size_t i = 0; i < path.size(); ++i) {
		const char c = path[i];
		cur.push_back(c);
		if (c == '\\' || c == '/' || i + 1 == path.size()) {
			if (cur.size() == 2 && cur[1] == ':') continue; // 跳过盘符
			if (cur == "\\" || cur == "/") continue;
			struct _stat st;
			if (_stat(cur.c_str(), &st) != 0) {
				_mkdir(cur.c_str());
			}
		}
	}
}

//bool savePointCloudToPly(const cv::Mat& xyz, const cv::Mat& P, const std::string& file_name)
//{
//	std::ofstream ply(file_name);
//
//	if (!ply)
//	{
//		std::cout << "open ply file error." << std::endl;
//		return false;
//	}
//
//	int element_count = 0;
//
//	for (int y = 0; y < P.rows; y += 1)
//	{
//		for (int x = 0; x < P.cols; x += 1)
//		{
//			if (P.at<uchar>(y, x) != 0)
//			{
//				element_count++;
//			}
//		}
//	}
//
//	std::cout << element_count << std::endl;
//	ply << "ply" << std::endl;
//	ply << "format ascii 1.0" << std::endl;
//	ply << "element vertex " << element_count << std::endl;
//	ply << "property float x" << std::endl;
//	ply << "property float y" << std::endl;
//	ply << "property float z" << std::endl;
//	ply << "end_header" << std::endl;
//
//	char str = ' ';
//	for (int y = 0; y < xyz.rows; y += 1)
//	{
//		for (int x = 0; x < xyz.cols; x += 1)
//		{
//			if (P.at<uchar>(y, x) != 0)
//			{
//				cv::Vec3f point = xyz.at<cv::Vec3f>(y, x);
//				ply << point[0] << " " << point[1] << " " << point[2] << std::endl;
//			}
//		}
//	}
//	ply.close();
//	return true;
//}
bool savePointCloudToPlyBinary(const cv::Mat& xyz, const cv::Mat& P, const std::string& file_name)
{
	// 1. 直接用 OpenCV 底层函数统计有效点数，去除双层 for 循环
	int element_count = cv::countNonZero(P);
	std::cout << "有效点数: " << element_count << std::endl;

	// 2. 使用微软安全版本的 fopen_s
	FILE* fp = nullptr;
	// fopen_s 的用法：第一个参数是文件指针的地址，返回值为 0 表示打开成功
	if (fopen_s(&fp, file_name.c_str(), "wb") != 0 || fp == nullptr)
	{
		std::cout << "open ply file error." << std::endl;
		return false;
	}

	// 3. 写入 PLY 二进制文件头
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n"); // 关键：声明为二进制
	fprintf(fp, "element vertex %d\n", element_count);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "end_header\n");

	// 4. 使用行指针 ptr 进行极速遍历，直接内存拷贝
	for (int y = 0; y < xyz.rows; y++)
	{
		const uchar* p_row = P.ptr<uchar>(y);
		const cv::Vec3f* xyz_row = xyz.ptr<cv::Vec3f>(y);

		for (int x = 0; x < xyz.cols; x++)
		{
			if (p_row[x] != 0)
			{
				// 直接将 3 个 float (12 byte) 的内存数据怼进硬盘，不经过任何字符转换！
				fwrite(&xyz_row[x], sizeof(float), 3, fp);
			}
		}
	}

	fclose(fp);
	return true;
}


static bool g_enable_gray_color = true;

static bool savePointCloudToPlyBinaryWithGray(const cv::Mat& xyz,
	const cv::Mat& P, const cv::Mat& gray_image,
	const std::string& file_name)
{
	if (xyz.empty() || P.empty() || xyz.size() != P.size() ||
		xyz.type() != CV_32FC3 || P.type() != CV_8UC1) return false;
	cv::Mat gray;
	if (g_enable_gray_color && !gray_image.empty()) {
		if (gray_image.type() != CV_8UC1 || gray_image.size() != xyz.size()) return false;
		gray = gray_image;
	} else gray = cv::Mat(xyz.size(), CV_8UC1, cv::Scalar(255));
	FILE* fp = nullptr;
	if (fopen_s(&fp, file_name.c_str(), "wb") != 0 || fp == nullptr) return false;
	const int count = cv::countNonZero(P);
	fprintf(fp, "ply\nformat binary_little_endian 1.0\nelement vertex %d\n", count);
	fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
	fprintf(fp, "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");
	for (int y = 0; y < xyz.rows; ++y) {
		const uchar* mask = P.ptr<uchar>(y);
		const uchar* color = gray.ptr<uchar>(y);
		const cv::Vec3f* point = xyz.ptr<cv::Vec3f>(y);
		for (int x = 0; x < xyz.cols; ++x) if (mask[x]) {
			fwrite(&point[x], sizeof(float), 3, fp);
			fwrite(&color[x], sizeof(uchar), 1, fp);
			fwrite(&color[x], sizeof(uchar), 1, fp);
			fwrite(&color[x], sizeof(uchar), 1, fp);
		}
	}
	fclose(fp);
	return true;
}void saveToTxt(const cv::Mat& xyz, const std::string& filename)
{
	// 辅助函数：将点云保存为 Matlab 可读取的纯文本 txt (X Y Z)
	std::ofstream out(filename);
	if (!out.is_open()) {
		std::cout << "无法创建文件: " << filename << std::endl;
		return;
	}

	for (int r = 0; r < xyz.rows; r++)
	{
		for (int c = 0; c < xyz.cols; c++)
		{
			cv::Vec3f p = xyz.at<cv::Vec3f>(r, c);
			// 过滤无效点 (假设 Z=0 或 Z非常大为无效，根据具体情况调整)
			if (p[2] > -10000 && p[2] < 10000 && abs(p[2]) > 0.0001)
			{
				out << p[0] << " " << p[1] << " " << p[2] << "\n";
			}
		}
	}
	out.close();
}
//系统标定
void calibration_v1()
{
	//图像文件夹路径
	//const std::string img_path = ".\\calib_imgs_20260523\\";
	const std::string img_pathL = "E:\\calib\\20261022_600\\L\\";
	const std::string img_pathR = "E:\\calib\\20261022_600\\R\\";//本地化
	//标定图像数量
	const int img_num =11;


	//图像加载并进行角点检测
	std::vector<std::vector<cv::Point2f>> left_image_points;
	std::vector<std::vector<cv::Point2f>> right_image_points;
	JSLAB_CircleGridDetector target_detect(cv::Size(11, 9), 3);//注意第二个参数是高斯卷积核

	for (int i = 1; i < img_num; i++)
	{
		//cv::Mat limg = cv::imread(img_path + "captureL_" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
		//cv::Mat rimg = cv::imread(img_path + "captureR_" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
		cv::Mat limg = cv::imread(img_pathL + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);//本地化
		cv::Mat rimg = cv::imread(img_pathR + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);//本地化
		if (limg.empty() || rimg.empty())
		{
			std::cout << "read image failed!" << std::endl;
			return;
		}

		std::vector<cv::Point2f> lcorners;
		std::vector<cv::Point2f> rcorners;
		cv::Mat ldraw_corners;
		cv::Mat rdraw_corners;

		if (!target_detect.detect(limg, lcorners, ldraw_corners) || !target_detect.detect(rimg, rcorners, rdraw_corners))
		{
			std::cout << "detect corners failed!" << std::endl;
			return;
		}

		cv::namedWindow("left_corners", cv::WINDOW_NORMAL);
		cv::namedWindow("right_corners", cv::WINDOW_NORMAL);
		cv::imshow("left_corners", ldraw_corners);
		cv::imshow("right_corners", rdraw_corners);
		cv::waitKey(1);
		left_image_points.push_back(lcorners);
		right_image_points.push_back(rcorners);
	}

	//构建标靶三维坐标
	std::vector<cv::Point3f> object_points;
	generateCircleGrid3DPoints(cv::Size(11, 9), 16, object_points);//这里改行列和圆心距;(列，行）

	//双目标定
	JSLAB_StereoCalib_V1 stereo_calib(cv::Size(5120, 5120));//这里改图片大小，例：2448*2048
	stereo_calib.setTargetImagePoints(left_image_points, right_image_points);
	stereo_calib.setTargetObjectPoints(object_points);
	double lrerr, lerr, rerr;
	stereo_calib.startCalib(lrerr, lerr, rerr);
	stereo_calib.visualizeCalibration();

	//保存标定结果
	stereo_calib.saveCalibResult("E:\\calib\\20261022_600\\calib_result_600FOV.xml");
}

/**
 * @brief 完整的混合标定流水线 (V2)
 * 包含：基础几何标定(或直接读取) -> 自动平面图像处理 -> 逐像素三轴坐标方程拟合
 */
void calibration_v2()
{
	std::cout << "\n==============================================" << std::endl;
	std::cout << "   [启动] 系统全自动混合标定流水线 (V2)" << std::endl;
	std::cout << "==============================================\n" << std::endl;

	// --- 【参数与开关配置】 ---
	const cv::Size sensor_size(5120, 5120);
	const int plane_num = 12;       // 用于拟合的平面数量

	// 是否使用已有的基础标定文件
	bool use_existing_calib = true;
	const std::string existing_calib_path = "./calib_result_20260427_200FOV.xml";

	// 路径配置
	const std::string plane_base_path = "F:\\20260428\\";

	// 实例化 V2 标定类
	JSLAB_StereoCalib_V2 stereo_calib_v2(sensor_size);

	// ==========================================================
	// 【阶段 1】：基础几何标定 (或直接加载)
	// ==========================================================
	if (use_existing_calib)
	{
		std::cout << ">>> 正在加载已有几何标定文件: " << existing_calib_path << std::endl;
		if (!stereo_calib_v2.loadBaseCalibration(existing_calib_path)) {
			std::cout << "[错误] 读取已有标定文件失败！请检查路径。" << std::endl;
			return;
		}
		std::cout << ">>> 基础内外参加载成功，跳过棋盘格检测！" << std::endl;
	}
	else
	{
		// 【原有的棋盘格标定逻辑】
		std::cout << ">>> 正在执行阶段 1：棋盘格几何标定..." << std::endl;
		const cv::Size board_size(11, 9);
		const double square_size = 8;
		const int grid_img_num = 20;
		const std::string grid_path = ".\\calib_imgs_20260427\\";

		std::vector<std::vector<cv::Point2f>> left_image_points, right_image_points;
		JSLAB_CircleGridDetector target_detect(board_size, 5);

		for (int i = 0; i < grid_img_num; i++) {
			cv::Mat limg = cv::imread(grid_path + "captureL_" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			cv::Mat rimg = cv::imread(grid_path + "captureR_" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			if (limg.empty() || rimg.empty()) continue;

			std::vector<cv::Point2f> lcorners, rcorners;
			cv::Mat ldraw, rdraw;
			if (target_detect.detect(limg, lcorners, ldraw) && target_detect.detect(rimg, rcorners, rdraw)) {
				left_image_points.push_back(lcorners);
				right_image_points.push_back(rcorners);
			}
		}

		std::vector<cv::Point3f> object_points;
		generateCircleGrid3DPoints(board_size, square_size, object_points);
		stereo_calib_v2.setTargetImagePoints(left_image_points, right_image_points);
		stereo_calib_v2.setTargetObjectPoints(object_points);

		double lrerr, lerr, rerr;
		stereo_calib_v2.startBaseCalibration(lrerr, lerr, rerr);
		stereo_calib_v2.visualizeCalibration(true);
	}

	// ==========================================================
	// 【数据桥梁】：初始化相移解码器与重建引擎
	// ==========================================================
	JSLAB_StereoCalibParams_V2 intermediate_params = stereo_calib_v2.getParams();
	JSLAB_StereoCalibParams_V1 v1_adapter;
	intermediate_params.camera_matrix[0].copyTo(v1_adapter.camera_matrix[0]);
	intermediate_params.camera_matrix[1].copyTo(v1_adapter.camera_matrix[1]);
	intermediate_params.dist_coeffs[0].copyTo(v1_adapter.dist_coeffs[0]);
	intermediate_params.dist_coeffs[1].copyTo(v1_adapter.dist_coeffs[1]);
	intermediate_params.R.copyTo(v1_adapter.R);
	intermediate_params.t.copyTo(v1_adapter.t);
	v1_adapter.image_size = intermediate_params.image_size;

	// 配置解码器 (8+8+8 相移模式)
	JSLAB_PhaseShiftDecoder::Config ps_config;
	ps_config.freq[0] = 107; ps_config.freq[1] = 100; ps_config.freq[2] = 94;
	ps_config.shift_num[0] = 8; ps_config.shift_num[1] = 8; ps_config.shift_num[2] = 8;
	ps_config.modulation_threshold = 1;
	auto phase_shift_decode = JSLAB_PhaseShiftDecoder::create(ps_config, JSLAB_PhaseShiftDecoder::Version::V2);

	// 配置重建器
	JSLAB_PhaseEpipolarReconstructor_V2::Config rebuild_config;
	rebuild_config.use_mls = false;
	JSLAB_PhaseEpipolarReconstructor_V2 base_rebuild(rebuild_config);
	base_rebuild.setCalibrationParams(v1_adapter);

	// ==========================================================
	// 【阶段 2】：多高度平面扫描与视差提取
	// ==========================================================
	std::cout << "\n>>> 正在执行阶段 2：参考平面视差采集..." << std::endl;

	for (int k = 1; k <= plane_num; k++)
	{
		std::cout << "--- 正在处理第 [" << k << "/" << plane_num << "] 个平面 ---" << std::endl;
		std::string pL = plane_base_path + "L\\" + std::to_string(k) + "\\";
		std::string pR = plane_base_path + "R\\" + std::to_string(k) + "\\";

		std::vector<cv::Mat> limgs, rimgs;
		for (int i = 0; i < 24; i++) // 8*3 = 24
		{
			cv::Mat imgL = cv::imread(pL + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			cv::Mat imgR = cv::imread(pR + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			if (imgL.empty() || imgR.empty()) break;
			limgs.push_back(imgL);
			rimgs.push_back(imgR);
		}

		// 【修正的Bug】：你原来写的是 < 25，因为循环是i<24，最多只会推入24张，导致所有平面全被跳过！这里已改为 24。
		if (limgs.size() < 24) {
			std::cout << "[警告] 平面 " << k << " 图像加载失败，跳过。" << std::endl;
			continue;
		}

		cv::Mat lphase, l_k, l_mod, rphase, r_k, r_mod;
		phase_shift_decode->setImages(limgs);
		phase_shift_decode->startCalculate(lphase);
		phase_shift_decode->getOrderMap(l_k);
		phase_shift_decode->getModulationMap(l_mod);

		phase_shift_decode->setImages(rimgs);
		phase_shift_decode->startCalculate(rphase);
		phase_shift_decode->getOrderMap(r_k);
		phase_shift_decode->getModulationMap(r_mod);

		base_rebuild.setData(lphase, rphase, l_k, r_k, l_mod, ps_config.freq[0]);

		cv::Mat temp_xyz, mask_P;
		base_rebuild.startCalculate(temp_xyz, mask_P);

		//cv::Mat current_disp = base_rebuild.getDisparityMap().clone();
		//stereo_calib_v2.addPlaneReferenceData(current_disp, mask_P);
	}

	// ==========================================================
	// 【阶段 3】：执行逐像素 3D 方程拟合并保存
	// ==========================================================
	std::cout << "\n>>> 正在执行阶段 3：逐像素三轴坐标方程拟合..." << std::endl;

	stereo_calib_v2.computePixelwiseDepthCalibration();

	// 保存包含几何参数与拟合系数的终极 XML 文件
	std::string final_save_path = "./calib_result_V2_Ultimate.xml";
	if (stereo_calib_v2.saveCalibResult(final_save_path)) {
		std::cout << "\n>>> [成功] 混合标定流水线执行完毕！" << std::endl;
		std::cout << ">>> 基础参数: " << final_save_path << std::endl;
		std::cout << ">>> 系数矩阵: " << "./calib_result_V2_Ultimate_rx/ry/rz.bin" << std::endl;
		std::cout << ">>> 该格式加载速度比纯 XML 快约 100 倍。" << std::endl;
	}
	else {
		std::cout << "\n>>> [错误] 标定文件保存失败！" << std::endl;
	}
}
//点云重建
void rebuild_v8()
{
	
	// 静态变量保存 ROI
	static std::vector<cv::RotatedRect> g_auto_rois;
	//const std::string base_path = "D:\\Pictures\\scan_imgs_20260430_0.1mmZ动态重复性\\位置3\\";
	// 【1】算法配置
	JSLAB_PhaseShiftDecoder::Config config;
	//config.freq[0] = 107; config.freq[1] = 100; config.freq[2] = 94;
	config.freq[0] = 121; config.freq[1] = 110; config.freq[2] = 120;
	config.shift_num[0] = 8; config.shift_num[1] = 8; config.shift_num[2] = 8;
	const double modulation_threshold = 0.3;
	config.modulation_threshold = modulation_threshold;
	const bool enable_hdr = false; // false: original phase decoding
	const bool enable_phase_preview = true; // Save/display maps, then wait before reconstruction.
	config.enable_hdr = enable_hdr; // 0 <= I < 255; keep multi-frequency unwrapping
	const bool enable_roi_selection = false; // true: enable ROI selection; false: reconstruct the full image
	const int output_format = 1; // 0: TXT, 1: binary PLY with grayscale color

	JSLAB_StereoCalibParams_V1 calib_params;
	if (!loadStereoCalibParams_V1("E:\\calib\\20261029\\calib_result_600FOV.xml", calib_params)) {
		std::cout << "读取标定文件失败！" << std::endl;
		return;
	}
	


	// 【2】外层初始化：复用内存 Buffer
	auto phase_shift_decode = JSLAB_PhaseShiftDecoder::create(config, JSLAB_PhaseShiftDecoder::Version::V2);
	JSLAB_PhaseEpipolarReconstructor_V2::Config rebuild_config;
	JSLAB_PhaseEpipolarReconstructor_V2 base_rebuild(rebuild_config);
	base_rebuild.setCalibrationParams(calib_params);
	const std::string base_path = "E:\\fpp\\环形工件亮\\";
	const std::string out_root = "E:\\output\\环形工件（无HDR）\\";
	ensure_dir(out_root);

	int nan = 1;
	// 【3】循环处理
	for (int k = 1; k <= 10; ++k)
	{
		
		std::cout << "\n>>> Processing Group: " << k << std::endl;
		std::string pathL = base_path + "L\\" + std::to_string(k) + "\\";
		std::string pathR = base_path + "R\\" + std::to_string(k) + "\\";
		std::vector<cv::Mat> limgs, rimgs;

		bool load_success = true;

		for (int i = 0; i < 24; i++)
		{
			cv::Mat limg = cv::imread(pathL + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			cv::Mat rimg = cv::imread(pathR + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);

			if (limg.empty() || rimg.empty()) {
				load_success = false;
				break;
			}
			limgs.push_back(limg);
			rimgs.push_back(rimg);
		}

		// 如果缺图，跳过本组，严防解码器崩溃
		if (!load_success) {
			std::cout << "第 " << k << " 组图像缺失或加载失败，跳过..." << std::endl;
			continue;
		}

		cv::Mat limg25 = cv::imread(pathL + "25.bmp", cv::IMREAD_GRAYSCALE);//哈哈//????

		cv::Mat limg25_rectified;

		base_rebuild.rectifyLeftImage(limg25, limg25_rectified);

		// 解码
		cv::Mat lphase, l_k, l_mod, rphase, r_k, r_mod;
		phase_shift_decode->setImages(limgs);
		phase_shift_decode->startCalculate(lphase);
		phase_shift_decode->getOrderMap(l_k);
		phase_shift_decode->getModulationMap(l_mod);

		phase_shift_decode->setImages(rimgs);
		phase_shift_decode->startCalculate(rphase);
		phase_shift_decode->getOrderMap(r_k);
		phase_shift_decode->getModulationMap(r_mod);		// 【4】初次设置全图，用于 ROI 自动探测
		if (enable_phase_preview) {
			previewDecodedStereoPhase(l_mod, r_mod, lphase, rphase, l_k, r_k,
			                          base_path + "phase preview/L/" + std::to_string(k),
			                          base_path + "phase preview/R/" + std::to_string(k),
			                          config.freq[0]);
		}
		base_rebuild.setData(lphase, rphase, l_k, r_k, l_mod, config.freq[0]);

		if (enable_roi_selection && g_auto_rois.empty()) {
			g_auto_rois = base_rebuild.detectTargetROIs(2, 20000 ,-1.0f);
			if (g_auto_rois.empty()) {
				std::cout << "未能检测到任何有效的阶梯块 ROI，跳过..." << std::endl;
				continue;
			}
		}
		if (!enable_roi_selection) {
			g_auto_rois.clear();
			g_auto_rois.push_back(cv::RotatedRect(
				cv::Point2f(l_mod.cols / 2.0f, l_mod.rows / 2.0f),
				cv::Size2f(static_cast<float>(l_mod.cols), static_cast<float>(l_mod.rows)),
				0.0f));
		}
		if (enable_roi_selection && !limg25.empty()&& nan++ == 1) {
			cv::Mat vis_white;
			// 将灰度白图转为彩色，以便画绿色的框
			cv::cvtColor(limg25, vis_white, cv::COLOR_GRAY2BGR);

			for (size_t i = 0; i < g_auto_rois.size(); ++i) {
				cv::Point2f vertices[4];
				g_auto_rois[i].points(vertices); // 获取 4 个角点

				// 画出四边形
				for (int j = 0; j < 4; j++) {
					cv::line(vis_white, vertices[j], vertices[(j + 1) % 4],
						cv::Scalar(0, 255, 0), 5, cv::LINE_AA); // 线宽设为10以便大图查看
				}

				// 标上 ID
				cv::putText(vis_white, "ROI_" + std::to_string(i), vertices[0],
					cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 3);
			}

			cv::namedWindow("ROI Check on White Image", cv::WINDOW_NORMAL);
			cv::imshow("ROI Check on White Image", vis_white);
			std::cout << ">>> 正在显示原图 ROI，按任意键继续重建..." << std::endl;
			cv::waitKey(0); // 阻塞一下，方便你确认框得准不准
			cv::destroyWindow("ROI Check on White Image");
		}
		// 【5】分块重建与保存
		std::string group_dir = out_root + std::to_string(k) + "/";
		_mkdir(group_dir.c_str());

		// 预分配画布，避免循环内内存申请
		cv::Mat canvas_mod = cv::Mat::zeros(l_mod.size(), l_mod.type());
		cv::Mat canvas_phase = cv::Mat::zeros(lphase.size(), lphase.type());

		for (int i = 0; i < g_auto_rois.size(); ++i)
		{
			std::cout << "--- Exporting Block " << i << " ---" << std::endl;

			// 每次循环前将画布彻底清零（擦黑）
			canvas_mod.setTo(0);
			canvas_phase.setTo(0);

			// 1. 获取四边形的 4 个角点
			cv::RotatedRect rbox = g_auto_rois[i];
			cv::Point2f pts_f[4];
			rbox.points(pts_f);

			// 将浮点角点转换为整型角点
			cv::Point pts_i[4];
			for (int j = 0; j < 4; j++) {
				pts_i[j] = cv::Point(cvRound(pts_f[j].x), cvRound(pts_f[j].y));
			}

			// 2. 创建一个全黑的掩膜，将四边形内部填涂为白色(255)
			cv::Mat poly_mask = cv::Mat::zeros(l_mod.size(), CV_8UC1);
			cv::fillConvexPoly(poly_mask, pts_i, 4, cv::Scalar(255));

			// 3. 核心截取：仅把原图中落在四边形区域内的数据，拷贝到画布上
			// OpenCV的 copyTo 支持掩膜，掩膜外的地方不会被复制，画布对应位置依然是 0
			l_mod.copyTo(canvas_mod, poly_mask);
			lphase.copyTo(canvas_phase, poly_mask);

			// 4. 传入仅含有倾斜四边形区域数据的画布，进行最终重建
			base_rebuild.setData(canvas_phase, rphase, l_k, r_k, canvas_mod, config.freq[0]);

			cv::Mat xyz, P;
			base_rebuild.startCalculate(xyz, P);

			if (output_format == 0)
			{
				char filename_buffer[64];
				sprintf_s(filename_buffer, "1_%06d.txt", i);
				std::string save_path = group_dir + std::string(filename_buffer);
				saveToTxt(xyz, save_path);
			}
			else if (output_format == 1)
			{
				char ply_name[64];
				sprintf_s(ply_name, "1_%06d.ply", i);
				savePointCloudToPlyBinaryWithGray(xyz, P, limg25_rectified,
					group_dir + std::string(ply_name));
			}

			xyz.release();
			P.release();
			poly_mask.release();
		}

		// 显式释放本组巨型矩阵，防止内存溢出
		lphase.release(); l_mod.release(); rphase.release();
		l_k.release(); r_k.release(); r_mod.release();
		canvas_mod.release(); canvas_phase.release();
		limgs.clear(); rimgs.clear();
		if (!limg25.empty()) limg25.release();
	}
	std::cout << "\nTask Finished Successfully!" << std::endl;
}

void rebuild_v8_multi_by_path(const std::string& base_path, const std::string& out_root, const std::vector<cv::RotatedRect>& batch_rois, const JSLAB_StereoCalibParams_V1& calib_params)
{
	// 使用当前文件夹对应的 ROI 处理该文件夹的 10 组数据
	std::vector<cv::RotatedRect> g_auto_rois = batch_rois;
	//const std::string base_path = "D:\\Pictures\\scan_imgs_20260430_0.1mmZ动态重复性\\位置3\\";
	// 【1】算法配置
	JSLAB_PhaseShiftDecoder::Config config;
	config.freq[0] = 107; config.freq[1] = 100; config.freq[2] = 94;
	config.shift_num[0] = 8; config.shift_num[1] = 8; config.shift_num[2] = 8;
	config.modulation_threshold = 1;


	// 【2】外层初始化：复用内存 Buffer
	auto phase_shift_decode = JSLAB_PhaseShiftDecoder::create(config, JSLAB_PhaseShiftDecoder::Version::V2);
	JSLAB_PhaseEpipolarReconstructor_V2::Config rebuild_config;
	JSLAB_PhaseEpipolarReconstructor_V2 base_rebuild(rebuild_config);
	base_rebuild.setCalibrationParams(calib_params);
	_mkdir(out_root.c_str());

	// 【3】循环处理
	for (int k = 1; k <= 10; ++k)
	{
		
		std::cout << "\n>>> Processing Group: " << k << std::endl;
		std::string pathL = base_path + "L\\" + std::to_string(k) + "\\";
		std::string pathR = base_path + "R\\" + std::to_string(k) + "\\";

		std::vector<cv::Mat> limgs, rimgs;

		bool load_success = true;

		for (int i = 0; i < 24; i++)
		{
			cv::Mat limg = cv::imread(pathL + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			cv::Mat rimg = cv::imread(pathR + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);

			if (limg.empty() || rimg.empty()) {
				load_success = false;
				break;
			}
			limgs.push_back(limg);
			rimgs.push_back(rimg);
		}

		// 如果缺图，跳过本组，严防解码器崩溃
		if (!load_success) {
			std::cout << "第 " << k << " 组图像缺失或加载失败，跳过..." << std::endl;
			continue;
		}

		// 解码
		cv::Mat limg25 = cv::imread(pathL + "25.bmp", cv::IMREAD_GRAYSCALE);
		 cv::Mat limg25_rectified;
		 base_rebuild.rectifyLeftImage(limg25, limg25_rectified);

		cv::Mat lphase, l_k, l_mod, rphase, r_k, r_mod;
		phase_shift_decode->setImages(limgs);
		phase_shift_decode->startCalculate(lphase);
		phase_shift_decode->getOrderMap(l_k);
		phase_shift_decode->getModulationMap(l_mod);

		phase_shift_decode->setImages(rimgs);
		phase_shift_decode->startCalculate(rphase);
		phase_shift_decode->getOrderMap(r_k);
		phase_shift_decode->getModulationMap(r_mod);

		// 【5】分块重建与保存
		std::string group_dir = out_root + std::to_string(k) + "/";
		_mkdir(group_dir.c_str());

		// 预分配画布，避免循环内内存申请
		cv::Mat canvas_mod = cv::Mat::zeros(l_mod.size(), l_mod.type());
		cv::Mat canvas_phase = cv::Mat::zeros(lphase.size(), lphase.type());

		for (int i = 0; i < g_auto_rois.size(); ++i)
		{
			std::cout << "--- Exporting Block " << i << " ---" << std::endl;

			// 每次循环前将画布彻底清零（擦黑）
			canvas_mod.setTo(0);
			canvas_phase.setTo(0);

			// 1. 获取四边形的 4 个角点
			cv::RotatedRect rbox = g_auto_rois[i];
			cv::Point2f pts_f[4];
			rbox.points(pts_f);

			// 将浮点角点转换为整型角点
			cv::Point pts_i[4];
			for (int j = 0; j < 4; j++) {
				pts_i[j] = cv::Point(cvRound(pts_f[j].x), cvRound(pts_f[j].y));
			}

			// 2. 创建一个全黑的掩膜，将四边形内部填涂为白色(255)
			cv::Mat poly_mask = cv::Mat::zeros(l_mod.size(), CV_8UC1);
			cv::fillConvexPoly(poly_mask, pts_i, 4, cv::Scalar(255));

			// 3. 核心截取：仅把原图中落在四边形区域内的数据，拷贝到画布上
			// OpenCV的 copyTo 支持掩膜，掩膜外的地方不会被复制，画布对应位置依然是 0
			l_mod.copyTo(canvas_mod, poly_mask);
			lphase.copyTo(canvas_phase, poly_mask);

			// 4. 传入仅含有倾斜四边形区域数据的画布，进行最终重建
			base_rebuild.setData(canvas_phase, rphase, l_k, r_k, canvas_mod, config.freq[0]);

			cv::Mat xyz, P;
			base_rebuild.startCalculate(xyz, P);

			// 分别保存为 Block_0.txt, Block_1.txt
			char filename_buffer[64];
			sprintf_s(filename_buffer, "1_%06d.txt", i);
			std::string save_path = group_dir + std::string(filename_buffer);
			char ply_name[64];
			sprintf_s(ply_name, "1_%06d.ply", i);
			savePointCloudToPlyBinaryWithGray(xyz, P, limg25_rectified, group_dir + std::string(ply_name));

			xyz.release();
			P.release();
			poly_mask.release();
		}

		// 显式释放本组巨型矩阵，防止内存溢出
		lphase.release(); l_mod.release(); rphase.release();
		l_k.release(); r_k.release(); r_mod.release();
		canvas_mod.release(); canvas_phase.release();
		limgs.clear(); rimgs.clear();
	}
	std::cout << "\nTask Finished Successfully!" << std::endl;
}

void 多次ROI框选()
{
	const std::string input_root = "E:\\fpp\\20260717_600L验收\\";
	const std::string output_root = "E:\\output\\20260717_600L验收\\";
	_mkdir(output_root.c_str());

	JSLAB_PhaseShiftDecoder::Config config;
	config.freq[0] = 107; config.freq[1] = 100; config.freq[2] = 94;
	config.shift_num[0] = 8; config.shift_num[1] = 8; config.shift_num[2] = 8;
	config.modulation_threshold = 1;

	JSLAB_StereoCalibParams_V1 calib_params;
	if (!loadStereoCalibParams_V1("E:\\calib\\20260716_热机\\calib_result_600FOV.xml", calib_params)) {
		std::cout << "读取标定文件失败！" << std::endl;
		return;
	}

	std::vector<std::vector<cv::RotatedRect>> all_folder_rois(5);

	// 先一次性完成 5 个文件夹的 ROI 框选
	for (int file_index = 1; file_index <= 5; ++file_index)
	{
		const std::string base_path = input_root + std::to_string(file_index) + "\\";
		const std::string roi_pathL = base_path + "L\\" + std::to_string(file_index) + "\\";
		const std::string roi_pathR = base_path + "R\\" + std::to_string(file_index) + "\\";

		auto phase_shift_decode = JSLAB_PhaseShiftDecoder::create(config, JSLAB_PhaseShiftDecoder::Version::V2);
		JSLAB_PhaseEpipolarReconstructor_V2::Config rebuild_config;
		JSLAB_PhaseEpipolarReconstructor_V2 base_rebuild(rebuild_config);
		base_rebuild.setCalibrationParams(calib_params);

		std::vector<cv::Mat> limgs, rimgs;
		bool load_success = true;
		for (int i = 0; i < 24; i++)
		{
			cv::Mat limg = cv::imread(roi_pathL + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			cv::Mat rimg = cv::imread(roi_pathR + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			if (limg.empty() || rimg.empty()) {
				load_success = false;
				break;
			}
			limgs.push_back(limg);
			rimgs.push_back(rimg);
		}
		if (!load_success) {
			std::cout << "第 " << file_index << " 个输入文件夹的 ROI 图像缺失或加载失败，跳过该文件夹。" << std::endl;
			continue;
		}

		cv::Mat lphase, l_k, l_mod, rphase, r_k, r_mod;
		phase_shift_decode->setImages(limgs);
		phase_shift_decode->startCalculate(lphase);
		phase_shift_decode->getOrderMap(l_k);
		phase_shift_decode->getModulationMap(l_mod);

		phase_shift_decode->setImages(rimgs);
		phase_shift_decode->startCalculate(rphase);
		phase_shift_decode->getOrderMap(r_k);
		phase_shift_decode->getModulationMap(r_mod);

		base_rebuild.setData(lphase, rphase, l_k, r_k, l_mod, config.freq[0]);
		std::vector<cv::RotatedRect> folder_rois = base_rebuild.detectTargetROIs(2, 20000, -1.0f);
		if (folder_rois.size() < 2) {
			std::cout << "第 " << file_index << " 个输入文件夹检测到的 ROI 数量不足 2 个，跳过该文件夹。当前数量: " << folder_rois.size() << std::endl;
			continue;
		}

		cv::Mat limg25 = cv::imread(roi_pathL + "25.bmp", cv::IMREAD_GRAYSCALE);
		if (!limg25.empty()) {
			cv::Mat vis_white;
			cv::cvtColor(limg25, vis_white, cv::COLOR_GRAY2BGR);
			for (size_t i = 0; i < folder_rois.size(); ++i) {
				cv::Point2f vertices[4];
				folder_rois[i].points(vertices);
				for (int j = 0; j < 4; j++) {
					cv::line(vis_white, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
				}
				cv::putText(vis_white, "ROI_" + std::to_string(i), vertices[0], cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 3);
			}
			cv::namedWindow("ROI Check on White Image", cv::WINDOW_NORMAL);
			cv::imshow("ROI Check on White Image", vis_white);
			std::cout << ">>> 第 " << file_index << " 个文件夹已框选 " << folder_rois.size() << " 个 ROI，按任意键继续框选下一组..." << std::endl;
			cv::waitKey(0);
			cv::destroyWindow("ROI Check on White Image");
		}

		all_folder_rois[file_index - 1] = folder_rois;

		lphase.release(); l_mod.release(); rphase.release();
		l_k.release(); r_k.release(); r_mod.release();
		limgs.clear(); rimgs.clear();
		if (!limg25.empty()) limg25.release();
	}

	std::cout << "\n>>> [多次ROI框选] 5 组 ROI 框选结束，开始依次重建..." << std::endl;

	// ROI 全部框选完成后，再依次执行 5 个文件夹的重建
	for (int file_index = 1; file_index <= 5; ++file_index)
	{
		if (all_folder_rois[file_index - 1].size() < 2) {
			std::cout << "第 " << file_index << " 个输入文件夹没有完整的 2 个 ROI，跳过重建。" << std::endl;
			continue;
		}

		const std::string base_path = input_root + std::to_string(file_index) + "\\";
		const std::string out_root = output_root + std::to_string(file_index) + "\\";

		std::cout << "\n>>> [多次ROI框选] 正在处理输入文件夹: " << base_path << std::endl;
		rebuild_v8_multi_by_path(base_path, out_root, all_folder_rois[file_index - 1], calib_params);
	}

	std::cout << "\n>>> [多次ROI框选] 5 个文件夹分别框选 2 个 ROI，共执行 50 组重建流程。" << std::endl;
}
/**
 * @brief 重建 V9 版本：集成逐像素多项式标定系数 (V2)
 * 功能：极速加载、自动 ROI 检测、高精度多项式拟合重建、传统 Q 阵保底回退
 */
void rebuild_v9()
{
	// 1. 静态变量保存 ROI 区域，确保序列重建时坐标系锁定
	static std::vector<cv::RotatedRect> g_auto_rois;

	// 【算法配置】
	JSLAB_PhaseShiftDecoder::Config config;
	config.freq[0] = 107; config.freq[1] = 100; config.freq[2] = 94;
	config.shift_num[0] = 8; config.shift_num[1] = 8; config.shift_num[2] = 8;
	config.modulation_threshold = 1;

	// 【核心：加载 V2 终极标定文件】
	// 注意：确保当前目录下存在 calib_result_V2_Ultimate.xml 及其对应的 _rx/ry/rz.bin 文件
	JSLAB_StereoCalibParams_V2 calib_params_v2;
	if (!loadStereoCalibParams_V2("./calib_result_V2_Ultimate.xml", calib_params_v2)) {
		std::cout << "[错误] 读取 V2 标定文件失败！请检查 XML 和 .bin 文件是否存在。" << std::endl;
		return;
	}
	std::cout << ">>> V2 混合标定参数(XML+BIN)加载成功，进入极速高精度模式。" << std::endl;

	// 【外层初始化：复用巨型内存 Buffer，断绝泄漏】
	auto phase_shift_decode = JSLAB_PhaseShiftDecoder::create(config, JSLAB_PhaseShiftDecoder::Version::V2);
	JSLAB_PhaseEpipolarReconstructor_V2::Config rebuild_config;
	rebuild_config.use_mls = false; // 如果需要追求拟合原始精度，建议设为 false；如果需要点云平整，设为 true

	JSLAB_PhaseEpipolarReconstructor_V2 base_rebuild(rebuild_config);

	// 【核心改变】：将 V2 参数注入重建器
	base_rebuild.setCalibrationParamsV2(calib_params_v2);

	const std::string base_path = "F:\\20260427\\ce\\";
	const std::string out_root = "./Output_Txt_V2/";
	_mkdir(out_root.c_str());

	int show_roi_flag = 1;

	// 【循环处理各组数据】
	for (int k = 1; k <= 10; ++k)
	{
		std::cout << "\n>>> [V2 重建] 正在处理第 " << k << " 组数据..." << std::endl;
		std::string pathL = base_path + "L\\" + std::to_string(k) + "\\";
		std::string pathR = base_path + "R\\" + std::to_string(k) + "\\";

		std::vector<cv::Mat> limgs, rimgs;
		bool load_success = true;

		for (int i = 0; i < 24; i++) // 8+8+8 = 24
		{
			cv::Mat limg = cv::imread(pathL + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			cv::Mat rimg = cv::imread(pathR + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
			if (limg.empty() || rimg.empty()) { load_success = false; break; }
			limgs.push_back(limg);
			rimgs.push_back(rimg);
		}

		if (!load_success) {
			std::cout << "第 " << k << " 组图像缺失，跳过..." << std::endl;
			continue;
		}

		// 额外读一张原白图用于 ROI 显示确认
		cv::Mat limg25 = cv::imread(pathL + "25.bmp", cv::IMREAD_GRAYSCALE);

		// 1. 相位解码
		cv::Mat lphase, l_k, l_mod, rphase, r_k, r_mod;
		phase_shift_decode->setImages(limgs);
		phase_shift_decode->startCalculate(lphase);
		phase_shift_decode->getOrderMap(l_k);
		phase_shift_decode->getModulationMap(l_mod);

		phase_shift_decode->setImages(rimgs);
		phase_shift_decode->startCalculate(rphase);
		phase_shift_decode->getOrderMap(r_k);
		phase_shift_decode->getModulationMap(r_mod);

		// 2. 初始全图设置，用于 ROI 探测
		base_rebuild.setData(lphase, rphase, l_k, r_k, l_mod, config.freq[0]);

		// 自动探测阶梯块 ROI (仅第一组执行)
		if (g_auto_rois.empty()) {
			g_auto_rois = base_rebuild.detectTargetROIs(2, 60000, 0.4f);
			if (g_auto_rois.empty()) {
				std::cout << "未能探测到 ROI，跳过本组..." << std::endl;
				continue;
			}
		}

		// 第一组显示 ROI 框选结果供确认
		if (!limg25.empty() && show_roi_flag == 1) {
			cv::Mat vis_white;
			cv::cvtColor(limg25, vis_white, cv::COLOR_GRAY2BGR);
			for (size_t i = 0; i < g_auto_rois.size(); ++i) {
				cv::Point2f vertices[4];
				g_auto_rois[i].points(vertices);
				for (int j = 0; j < 4; j++)
					cv::line(vis_white, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
				cv::putText(vis_white, "ROI_" + std::to_string(i), vertices[0], cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 3);
			}
			cv::namedWindow("ROI Check (V2 Mode)", cv::WINDOW_NORMAL);
			cv::imshow("ROI Check (V2 Mode)", vis_white);
			std::cout << ">>> 正在显示 ROI，按任意键开始高精度重建..." << std::endl;
			cv::waitKey(0);
			cv::destroyWindow("ROI Check (V2 Mode)");
			show_roi_flag = 0;
		}

		// 3. 分块截取并执行 V2 重建
		std::string group_dir = out_root + std::to_string(k) + "/";
		_mkdir(group_dir.c_str());

		cv::Mat canvas_mod = cv::Mat::zeros(l_mod.size(), l_mod.type());
		cv::Mat canvas_phase = cv::Mat::zeros(lphase.size(), lphase.type());

		for (int i = 0; i < g_auto_rois.size(); ++i)
		{
			std::cout << "--- 正在处理区域 " << i << " ---" << std::endl;
			canvas_mod.setTo(0); canvas_phase.setTo(0);

			cv::RotatedRect rbox = g_auto_rois[i];
			cv::Point2f pts_f[4];
			rbox.points(pts_f);
			cv::Point pts_i[4];
			for (int j = 0; j < 4; j++) pts_i[j] = cv::Point(cvRound(pts_f[j].x), cvRound(pts_f[j].y));

			cv::Mat poly_mask = cv::Mat::zeros(l_mod.size(), CV_8UC1);
			cv::fillConvexPoly(poly_mask, pts_i, 4, cv::Scalar(255));

			l_mod.copyTo(canvas_mod, poly_mask);
			lphase.copyTo(canvas_phase, poly_mask);

			// 注入局部数据
			base_rebuild.setData(canvas_phase, rphase, l_k, r_k, canvas_mod, config.freq[0]);

			// 【核心：调用新写的 startCalculate_v2 接口】
			cv::Mat xyz, P;
			base_rebuild.startCalculate_v2(xyz, P);

			// 保存 TXT
			char filename_buffer[64];
			sprintf_s(filename_buffer, "1_%06d.txt", i);
			std::string save_path = group_dir + std::string(filename_buffer);
			saveToTxt(xyz, save_path);

			xyz.release(); P.release(); poly_mask.release();
		}

		// 显式释放本组巨型矩阵，防止长时间运行内存溢出
		lphase.release(); l_mod.release(); rphase.release();
		l_k.release(); r_k.release(); r_mod.release();
		canvas_mod.release(); canvas_phase.release();
		limgs.clear(); rimgs.clear();
		if (!limg25.empty()) limg25.release();
	}

	std::cout << "\n>>> [完成] 所有组高精度重建任务处理完毕。" << std::endl;
}

int main()
{
	//系统标定
//	calibration_v1();
	//点云重建
	rebuild_v8();

//savePointCloudToPlyBinaryWithGray
//	多次ROI框选();
//return 0;
}
