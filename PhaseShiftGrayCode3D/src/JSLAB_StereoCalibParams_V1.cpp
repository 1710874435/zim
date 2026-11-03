#include "JSLAB_StereoCalibParams_V1.h"

bool loadStereoCalibParams_V1(const std::string& filename, JSLAB_StereoCalibParams_V1& params)
{
	// 以 OpenCV FileStorage 格式读取 V1 XML/YAML 标定文件。
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;

	// 文件字段名必须与 JSLAB_StereoCalib_V1::saveCalibResult() 完全一致。
	fs["LeftCameraMatrix"]            >> params.camera_matrix[0];
	fs["LeftDistortionCoefficients"]  >> params.dist_coeffs[0];
	fs["RightCameraMatrix"]           >> params.camera_matrix[1];
	fs["RightDistortionCoefficients"] >> params.dist_coeffs[1];
	fs["RotationMatrix"]              >> params.R;
	fs["TranslationVector"]           >> params.t;
	fs["ImageSize"]                   >> params.image_size;
	fs.release();
	return true;
}
