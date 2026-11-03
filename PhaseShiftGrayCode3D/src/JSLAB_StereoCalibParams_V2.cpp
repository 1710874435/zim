#include "JSLAB_StereoCalibParams_V2.h"
#include <fstream>

bool JSLAB_StereoCalibParams_V2::loadMatBinary(const std::string& filename, cv::Mat& mat) {
	// 二进制文件布局：type、rows、cols 三个 int，随后是连续矩阵数据。
	std::ifstream fs(filename, std::ios::binary);
	if (!fs.is_open()) return false;
	int type, rows, cols;
	fs.read((char*)&type, sizeof(int));
	fs.read((char*)&rows, sizeof(int));
	fs.read((char*)&cols, sizeof(int));
	// 根据文件头恢复 OpenCV 矩阵类型和尺寸，再一次性读入全部像素。
	mat.create(rows, cols, type);
	fs.read((char*)mat.data, mat.step[0] * mat.rows);
	fs.close();
	return true;
}

bool loadStereoCalibParams_V2(const std::string& filename, JSLAB_StereoCalibParams_V2& params)
{
	// XML 保存基础双目几何参数；逐像素系数保存于三个同名二进制文件。
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;

	fs["LeftCameraMatrix"]				>> params.camera_matrix[0];
	fs["LeftDistortionCoefficients"]    >> params.dist_coeffs[0];
	fs["RightCameraMatrix"]				>> params.camera_matrix[1];
	fs["RightDistortionCoefficients"]   >> params.dist_coeffs[1];
	fs["RotationMatrix"]				>> params.R;
	fs["TranslationVector"]			    >> params.t;
	fs["ImageSize"]					    >> params.image_size;
	fs.release();

	// 由 XML 路径推导 X、Y、Z 三个坐标方程的系数文件名。
	std::string base_path = filename.substr(0, filename.find_last_of('.'));
	bool b1 = JSLAB_StereoCalibParams_V2::loadMatBinary(base_path + "_rx.bin", params.xyz_calib_coeffs[0]);
	bool b2 = JSLAB_StereoCalibParams_V2::loadMatBinary(base_path + "_ry.bin", params.xyz_calib_coeffs[1]);
	bool b3 = JSLAB_StereoCalibParams_V2::loadMatBinary(base_path + "_rz.bin", params.xyz_calib_coeffs[2]);

	// 三个坐标分量的系数全部读取成功后，V2 参数才完整有效。
	return b1 && b2 && b3;;
}