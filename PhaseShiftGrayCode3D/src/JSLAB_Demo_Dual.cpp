//#include <iostream>
//#include <JSLAB_CircleGridDetector.h>
//#include <JSLAB_StereoCalib_V1.h>
//#include <JSLAB_PhaseShiftDecoder.h>
//#include <JSLAB_PhaseShiftDecoder_V2_GPU.h>
//#include <JSLAB_PhaseEpipolarReconstructor_V2.h>
//#include <JSLAB_PhaseEpipolarReconstructor_V2_GPU.h>
//#include <JSLAB_StereoCalibParams_V1.h>
//#include <fstream>
//#include <chrono>
//#include <cmath>
//
//static const std::string DATA_DIR = "E:/code/PhaseShiftGrayCode3D_multi_fre/bin/Release/demo重建数据/scan_imgs_20260430_0.1mm动态重复性_50fov_频率107_100_94/";
//static const std::string CALIB_RESULT = DATA_DIR + "calib_result.xml";
//
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
//
//void calibration()
//{
//	const std::string img_path = DATA_DIR + "calib_imgs_20260611/";
//	const int img_num = 5;
//
//	std::vector<std::vector<cv::Point2f>> left_image_points;
//	std::vector<std::vector<cv::Point2f>> right_image_points;
//	JSLAB_CircleGridDetector target_detect(cv::Size(11, 9), 5,
//		100000, 1000, true, true, 255, 30, 0.6f, 1.0f, true);
//
//	for (int i = 0; i < img_num; i++)
//	{
//		cv::Mat limg = cv::imread(img_path + "L/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		cv::Mat rimg = cv::imread(img_path + "R/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		if (limg.empty() || rimg.empty())
//		{
//			std::cout << "read image failed: " << i + 1 << std::endl;
//			return;
//		}
//
//		std::vector<cv::Point2f> lcorners;
//		std::vector<cv::Point2f> rcorners;
//		cv::Mat ldraw_corners;
//		cv::Mat rdraw_corners;
//
//		if (!target_detect.detect(limg, lcorners, ldraw_corners) || !target_detect.detect(rimg, rcorners, rdraw_corners))
//		{
//			std::cout << "detect corners failed: " << i + 1 << std::endl;
//			continue;
//		}
//
//		cv::namedWindow("left_corners", cv::WINDOW_NORMAL);
//		cv::namedWindow("right_corners", cv::WINDOW_NORMAL);
//		cv::imshow("left_corners", ldraw_corners);
//		cv::imshow("right_corners", rdraw_corners);
//		cv::waitKey(1);
//		left_image_points.push_back(lcorners);
//		right_image_points.push_back(rcorners);
//	}
//
//	std::vector<cv::Point3f> object_points;
//	generateCircleGrid3DPoints(cv::Size(11, 9), 1.5, object_points);
//
//	JSLAB_StereoCalib_V1 stereo_calib(cv::Size(5120, 5120));
//	stereo_calib.setTargetImagePoints(left_image_points, right_image_points);
//	stereo_calib.setTargetObjectPoints(object_points);
//	double lrerr, lerr, rerr;
//	stereo_calib.startCalib(lrerr, lerr, rerr);
//	std::cout << "stereo err=" << lrerr << "  left err=" << lerr << "  right err=" << rerr << std::endl;
//
//	stereo_calib.saveCalibResult(CALIB_RESULT);
//	std::cout << "标定完成，结果已保存到: " << CALIB_RESULT << std::endl;
//}
//
//void rebuild_v2()
//{
//	const std::string img_path = DATA_DIR ;
//
//	JSLAB_PhaseShiftDecoder::Config config;
//	config.freq[0] = 107;
//	config.freq[1] = 100;
//	config.freq[2] = 94;
//	config.shift_num[0] = 8;
//	config.shift_num[1] = 8;
//	config.shift_num[2] = 8;
//	config.modulation_threshold = 1.0;
//	config.preblur_sigma = 3.0;
//	config.avg_three_phase = true;
//	config.post_smooth = true;
//
//	std::vector<cv::Mat> limgs, rimgs;
//	const int total_imgs = config.shift_num[0] + config.shift_num[1] + config.shift_num[2];
//	for (int i = 0; i < total_imgs; i++)
//	{
//		cv::Mat limg = cv::imread(img_path + "L/1/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		cv::Mat rimg = cv::imread(img_path + "R/1/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		if (limg.empty() || rimg.empty())
//		{
//			std::cout << "read image failed: " << i + 1 << std::endl;
//			return;
//		}
//		limgs.push_back(limg);
//		rimgs.push_back(rimg);
//	}
//
//	JSLAB_StereoCalibParams_V1 calib_params;
//	if (!loadStereoCalibParams_V1(CALIB_RESULT, calib_params))
//	{
//		std::cout << "read calib params failed!" << std::endl;
//		return;
//	}
//
//	std::cout << "图像读取完成，开始解码..." << std::endl;
//	auto phase_shift_decode = JSLAB_PhaseShiftDecoder::create(config, JSLAB_PhaseShiftDecoder::Version::V2);
//
//	JSLAB_PhaseEpipolarReconstructor_V2::Config rebuild_config;
//	rebuild_config.min_depth = 0;
//	rebuild_config.max_depth = 1000;
//	rebuild_config.min_phase_diff = 0.03;
//	rebuild_config.max_phase_diff = 0.4;
//	rebuild_config.radius_search = 0.05f;
//	rebuild_config.radius_min_neighbors = 5;
//	rebuild_config.radius_window_size = 5;
//	rebuild_config.use_mls = false;
//	rebuild_config.mls_window_size = 50;
//	rebuild_config.mls_spatial_sigma = 0.05f;
//	rebuild_config.use_gpu_mls = false;
//	rebuild_config.mls_max_radius = 0.0f;
//	rebuild_config.mls_iterations = 2;
//	JSLAB_PhaseEpipolarReconstructor_V2 reconstructor(rebuild_config);
//	reconstructor.setCalibrationParams(calib_params);
//
//	using clk = std::chrono::high_resolution_clock;
//	auto ms = [](clk::time_point a, clk::time_point b)
//		{ return std::chrono::duration<double, std::milli>(b - a).count(); };
//
//	auto t_start = clk::now();
//
//	cv::Mat lphase, l_k, l_mod;
//	phase_shift_decode->setImages(limgs);
//	phase_shift_decode->startCalculate(lphase);
//	phase_shift_decode->getOrderMap(l_k);
//	phase_shift_decode->getModulationMap(l_mod);
//
//	cv::Mat rphase, r_k;
//	phase_shift_decode->setImages(rimgs);
//	phase_shift_decode->startCalculate(rphase);
//	phase_shift_decode->getOrderMap(r_k);
//	auto t_decode = clk::now();
//
//	std::cout << "解码完成，开始重建..." << std::endl;
//
//	reconstructor.setData(lphase, rphase, l_k, r_k, l_mod, config.freq[0]);
//	auto t_setdata = clk::now();
//
//	cv::Mat xyz, P;
//	reconstructor.startCalculate(xyz, P);
//	auto t_end = clk::now();
//
//	std::cout << "[CPU][Timing] decode       = " << ms(t_start, t_decode) << " ms" << std::endl;
//	std::cout << "[CPU][Timing] setData/remap= " << ms(t_decode, t_setdata) << " ms" << std::endl;
//	std::cout << "[CPU][Timing] reconstruct  = " << ms(t_setdata, t_end) << " ms" << std::endl;
//	std::cout << "[CPU][Timing] TOTAL        = " << ms(t_start, t_end) << " ms" << std::endl;
//	std::cout << "重建完成，正在保存点云到ply文件..." << std::endl;
//
//	const std::string ply_path = DATA_DIR + "rebuild.ply";
//	savePointCloudToPly(xyz, P, ply_path);
//	std::cout << "点云文件保存完成: " << ply_path << std::endl;
//}
//
//void rebuild_v2_gpu()
//{
//	const std::string img_path = DATA_DIR ;
//
//	JSLAB_PhaseShiftDecoder::Config config;
//	config.freq[0] = 107;
//	config.freq[1] = 100;
//	config.freq[2] = 94;
//	config.shift_num[0] = 8;
//	config.shift_num[1] = 8;
//	config.shift_num[2] = 8;
//	config.modulation_threshold = 1.0;
//	config.preblur_sigma = 3.0;
//	config.avg_three_phase = true;
//	config.post_smooth = true;
//
//	std::vector<cv::Mat> limgs, rimgs;
//	const int total_imgs = config.shift_num[0] + config.shift_num[1] + config.shift_num[2];
//	for (int i = 0; i < total_imgs; i++)
//	{
//		cv::Mat limg = cv::imread(img_path + "L/1/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		cv::Mat rimg = cv::imread(img_path + "R/1/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		if (limg.empty() || rimg.empty())
//		{
//			std::cout << "read image failed: " << i + 1 << std::endl;
//			return;
//		}
//		limgs.push_back(limg);
//		rimgs.push_back(rimg);
//	}
//
//	JSLAB_StereoCalibParams_V1 calib_params;
//	if (!loadStereoCalibParams_V1(CALIB_RESULT, calib_params))
//	{
//		std::cout << "read calib params failed!" << std::endl;
//		return;
//	}
//
//	std::cout << "[GPU] 图像读取完成，开始解码..." << std::endl;
//	JSLAB_PhaseShiftDecoderV2_GPU phase_shift_decode(config);
//
//	JSLAB_PhaseEpipolarReconstructor_V2_GPU::Config rebuild_config;
//	rebuild_config.min_depth = 0;
//	rebuild_config.max_depth = 1000;
//	rebuild_config.min_phase_diff = 0.03;
//	rebuild_config.max_phase_diff = 0.4;
//	rebuild_config.radius_search = 0.05f;
//	rebuild_config.radius_min_neighbors = 5;
//	rebuild_config.radius_window_size = 5;
//	rebuild_config.use_mls = false;
//	rebuild_config.mls_window_size = 50;
//	rebuild_config.mls_spatial_sigma = 0.05f;
//	rebuild_config.use_gpu_mls = true;
//	rebuild_config.mls_max_radius = 0.0f;
//	rebuild_config.mls_iterations = 2;
//	JSLAB_PhaseEpipolarReconstructor_V2_GPU reconstructor(rebuild_config);
//	reconstructor.setCalibrationParams(calib_params);
//
//	using clk = std::chrono::high_resolution_clock;
//	auto ms = [](clk::time_point a, clk::time_point b)
//		{ return std::chrono::duration<double, std::milli>(b - a).count(); };
//
//	 //Initialize the CUDA context now so its one-time setup is not charged to
//	 //the decode timing below (amortized in real batch processing anyway).
//	JSLAB_DecodePhase_GPU_Warmup();
//
//	auto t_start = clk::now();
//
//	cv::Mat lphase, l_k, l_mod;
//	phase_shift_decode.setImages(limgs);
//	phase_shift_decode.startCalculate(lphase);
//	phase_shift_decode.getOrderMap(l_k);
//	phase_shift_decode.getModulationMap(l_mod);
//
//	cv::Mat rphase, r_k;
//	phase_shift_decode.setImages(rimgs);
//	phase_shift_decode.startCalculate(rphase);
//	phase_shift_decode.getOrderMap(r_k);
//	auto t_decode = clk::now();
//
//	std::cout << "[GPU] 解码完成，开始重建..." << std::endl;
//
//	reconstructor.setData(lphase, rphase, l_k, r_k, l_mod, config.freq[0]);
//	auto t_setdata = clk::now();
//
//	cv::Mat xyz, P;
//	reconstructor.startCalculate(xyz, P);
//	auto t_end = clk::now();
//
//	std::cout << "[GPU][Timing] decode       = " << ms(t_start, t_decode) << " ms" << std::endl;
//	std::cout << "[GPU][Timing] setData/remap= " << ms(t_decode, t_setdata) << " ms" << std::endl;
//	std::cout << "[GPU][Timing] reconstruct  = " << ms(t_setdata, t_end) << " ms" << std::endl;
//	std::cout << "[GPU][Timing] TOTAL        = " << ms(t_start, t_end) << " ms" << std::endl;
//	std::cout << "[GPU] 重建完成，正在保存点云到ply文件..." << std::endl;
//
//	const std::string ply_path = DATA_DIR + "rebuild_gpu.ply";
//	savePointCloudToPly(xyz, P, ply_path);
//	std::cout << "[GPU] 点云文件保存完成: " << ply_path << std::endl;
//}
//
// //===========================================================================
// //Diagnostic: compare CPU vs GPU pipelines stage by stage.
// //===========================================================================
//
// //Compare two single-channel float/double maps (reports max + mean abs diff).
//static void compareScalarMaps(const std::string& tag, const cv::Mat& a_in, const cv::Mat& b_in)
//{
//	cv::Mat a, b;
//	a_in.convertTo(a, CV_64F);
//	b_in.convertTo(b, CV_64F);
//
//	cv::Mat diff;
//	cv::absdiff(a, b, diff);
//
//	double max_diff = 0.0;
//	cv::minMaxLoc(diff, nullptr, &max_diff);
//	double mean_diff = cv::mean(diff)[0];
//	int    nz = cv::countNonZero(diff > 1e-6);
//
//	std::cout << "[Compare] " << tag
//		<< "  MaxDiff=" << max_diff
//		<< "  MeanDiff=" << mean_diff
//		<< "  DiffPixels(>1e-6)=" << nz << std::endl;
//}
//
// //Compare two integer order maps (reports number of differing pixels).
//static void compareOrderMaps(const std::string& tag, const cv::Mat& a, const cv::Mat& b)
//{
//	cv::Mat neq = (a != b);
//	int n_diff = cv::countNonZero(neq);
//	double pct = 100.0 * n_diff / (double)(a.rows * a.cols);
//	std::cout << "[Compare] " << tag
//		<< "  DiffPixels=" << n_diff
//		<< " (" << pct << "%)" << std::endl;
//}
//
//void compare_cpu_vs_gpu()
//{
//	const std::string img_path = DATA_DIR ;
//
//	// ---- Decoder config (shared by CPU and GPU) ----
//	JSLAB_PhaseShiftDecoder::Config config;
//	config.freq[0] = 107;
//	config.freq[1] = 100;
//	config.freq[2] = 94;
//	config.shift_num[0] = 8;
//	config.shift_num[1] = 8;
//	config.shift_num[2] = 8;
//	config.modulation_threshold = 1.0;
//	config.preblur_sigma = 3.0;
//	config.avg_three_phase = true;
//	config.post_smooth = true;
//
//	// ---- Load images ----
//	std::vector<cv::Mat> limgs, rimgs;
//	const int total_imgs = config.shift_num[0] + config.shift_num[1] + config.shift_num[2];
//	for (int i = 0; i < total_imgs; i++)
//	{
//		cv::Mat limg = cv::imread(img_path + "L/1/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		cv::Mat rimg = cv::imread(img_path + "R/1/" + std::to_string(i + 1) + ".bmp", cv::IMREAD_GRAYSCALE);
//		if (limg.empty() || rimg.empty())
//		{
//			std::cout << "read image failed: " << i + 1 << std::endl;
//			return;
//		}
//		limgs.push_back(limg);
//		rimgs.push_back(rimg);
//	}
//
//	JSLAB_StereoCalibParams_V1 calib_params;
//	if (!loadStereoCalibParams_V1(CALIB_RESULT, calib_params))
//	{
//		std::cout << "read calib params failed!" << std::endl;
//		return;
//	}
//
//	/* =======================================================================
//	 Stage 1: phase decoding comparison
//	 =======================================================================*/
//	std::cout << "\n========== Stage 1: Phase Decoding ==========" << std::endl;
//	
//	// ---- CPU decoder ----
//	auto decode_cpu = JSLAB_PhaseShiftDecoder::create(config, JSLAB_PhaseShiftDecoder::Version::V2);
//	cv::Mat lphase_cpu, l_k_cpu, l_mod_cpu, rphase_cpu, r_k_cpu;
//	decode_cpu->setImages(limgs);
//	decode_cpu->startCalculate(lphase_cpu);
//	decode_cpu->getOrderMap(l_k_cpu);
//	decode_cpu->getModulationMap(l_mod_cpu);
//	decode_cpu->setImages(rimgs);
//	decode_cpu->startCalculate(rphase_cpu);
//	decode_cpu->getOrderMap(r_k_cpu);
//
//	 //---- GPU decoder ----
//	JSLAB_PhaseShiftDecoderV2_GPU decode_gpu(config);
//	cv::Mat lphase_gpu, l_k_gpu, l_mod_gpu, rphase_gpu, r_k_gpu;
//	decode_gpu.setImages(limgs);
//	decode_gpu.startCalculate(lphase_gpu);
//	decode_gpu.getOrderMap(l_k_gpu);
//	decode_gpu.getModulationMap(l_mod_gpu);
//	decode_gpu.setImages(rimgs);
//	decode_gpu.startCalculate(rphase_gpu);
//	decode_gpu.getOrderMap(r_k_gpu);
//
//	compareScalarMaps("Left  Phase Map (CV_64F)", lphase_cpu, lphase_gpu);
//	compareScalarMaps("Right Phase Map (CV_64F)", rphase_cpu, rphase_gpu);
//	compareOrderMaps("Left  Order Map (CV_32S)", l_k_cpu, l_k_gpu);
//	compareOrderMaps("Right Order Map (CV_32S)", r_k_cpu, r_k_gpu);
//
//	/* =======================================================================
//	 Stage 2 + 3: reconstruction comparison using IDENTICAL CPU-decoded maps
//	 =======================================================================*/
//	std::cout << "\n========== Stage 2/3: Reconstruction (identical CPU inputs) ==========" << std::endl;
//
//	JSLAB_PhaseEpipolarReconstructor_V2::Config recon_cfg_cpu;
//	recon_cfg_cpu.min_depth = 0;
//	recon_cfg_cpu.max_depth = 1000;
//	recon_cfg_cpu.radius_search = 0.05f;
//	recon_cfg_cpu.radius_min_neighbors = 5;
//	recon_cfg_cpu.radius_window_size = 5;
//	recon_cfg_cpu.use_disp_median_filter = true;
//	recon_cfg_cpu.use_mls = false;   // keep off so the comparison stays geometric
//
//	JSLAB_PhaseEpipolarReconstructor_V2_GPU::Config recon_cfg_gpu;
//	recon_cfg_gpu.min_depth = 0;
//	recon_cfg_gpu.max_depth = 1000;
//	recon_cfg_gpu.radius_search = 0.05f;
//	recon_cfg_gpu.radius_min_neighbors = 5;
//	recon_cfg_gpu.radius_window_size = 5;
//	recon_cfg_gpu.use_disp_median_filter = true;
//	recon_cfg_gpu.use_mls = false;
//
//	JSLAB_PhaseEpipolarReconstructor_V2     recon_cpu(recon_cfg_cpu);
//	JSLAB_PhaseEpipolarReconstructor_V2_GPU recon_gpu(recon_cfg_gpu);
//	recon_cpu.setCalibrationParams(calib_params);
//	recon_gpu.setCalibrationParams(calib_params);
//
//	// Feed the SAME CPU-decoded maps into both reconstructors.
//	recon_cpu.setData(lphase_cpu, rphase_cpu, l_k_cpu, r_k_cpu, l_mod_cpu, config.freq[0]);
//	recon_gpu.setData(lphase_cpu, rphase_cpu, l_k_cpu, r_k_cpu, l_mod_cpu, config.freq[0]);
//
//	cv::Mat xyz_cpu, P_cpu, xyz_gpu, P_gpu;
//	recon_cpu.startCalculate(xyz_cpu, P_cpu);
//	recon_gpu.startCalculate(xyz_gpu, P_gpu);
//
//	 //---- Stage 2: disparity comparison ----
//	std::cout << "\n----- Stage 2: Disparity -----" << std::endl;
//	cv::Mat disp_cpu, disp_gpu;
//	recon_cpu.getDisparityMap(disp_cpu);   // CV_64FC1
//	recon_gpu.getDisparityMap(disp_gpu);   // CV_32FC1
//
//	if (disp_cpu.empty() || disp_gpu.empty())
//	{
//		std::cout << "[Compare] disparity map empty (cpu_empty=" << disp_cpu.empty()
//			<< " gpu_empty=" << disp_gpu.empty() << ")" << std::endl;
//	}
//	else
//	{
//		int nz_cpu = cv::countNonZero(cv::abs(disp_cpu) > 1e-5);
//		cv::Mat disp_gpu64; disp_gpu.convertTo(disp_gpu64, CV_64F);
//		int nz_gpu = cv::countNonZero(cv::abs(disp_gpu64) > 1e-5);
//		std::cout << "[Compare] Disparity valid pixels  CPU=" << nz_cpu
//			<< "  GPU=" << nz_gpu << std::endl;
//		compareScalarMaps("Disparity Map", disp_cpu, disp_gpu64);
//	}
//
//	// ---- Stage 3: point cloud + mask comparison ----
//	std::cout << "\n----- Stage 3: Point Cloud -----" << std::endl;
//	int valid_cpu = cv::countNonZero(P_cpu);
//	int valid_gpu = cv::countNonZero(P_gpu);
//
//	cv::Mat both = (P_cpu != 0) & (P_gpu != 0);
//	int overlap = cv::countNonZero(both);
//
//	double sum_err = 0.0, max_err = 0.0;
//	int    cnt = 0;
//	for (int y = 0; y < xyz_cpu.rows; ++y)
//	{
//		const cv::Vec3f* a = xyz_cpu.ptr<cv::Vec3f>(y);
//		const cv::Vec3f* b = xyz_gpu.ptr<cv::Vec3f>(y);
//		const uchar* m = both.ptr<uchar>(y);
//		for (int x = 0; x < xyz_cpu.cols; ++x)
//		{
//			if (m[x] == 0)
//				continue;
//			double dx = (double)a[x][0] - b[x][0];
//			double dy = (double)a[x][1] - b[x][1];
//			double dz = (double)a[x][2] - b[x][2];
//			double e = std::sqrt(dx * dx + dy * dy + dz * dz);
//			sum_err += e;
//			if (e > max_err) max_err = e;
//			++cnt;
//		}
//	}
//	double mean_err = (cnt > 0) ? sum_err / cnt : 0.0;
//
//	std::cout << "[Compare] Valid points  CPU=" << valid_cpu
//		<< "  GPU=" << valid_gpu
//		<< "  Overlap=" << overlap << std::endl;
//	std::cout << "[Compare] Point cloud geometric error over overlap:"
//		<< "  MeanErr=" << mean_err
//		<< "  MaxErr=" << max_err
//		<< "  (n=" << cnt << ")" << std::endl;
//
//	savePointCloudToPly(xyz_cpu, P_cpu, DATA_DIR + "compare_cpu.ply");
//	savePointCloudToPly(xyz_gpu, P_gpu, DATA_DIR + "compare_gpu.ply");
//	std::cout << "[Compare] Saved compare_cpu.ply / compare_gpu.ply" << std::endl;
//}
//
//int main()
//{
//	calibration();
//	//rebuild_v2();
//	//rebuild_v2_gpu();
//	//compare_cpu_vs_gpu();
//	//return 0;
//}