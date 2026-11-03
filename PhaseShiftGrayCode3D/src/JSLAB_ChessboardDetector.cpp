#include "JSLAB_ChessboardDetector.h"

JSLAB_ChessboardDetector::JSLAB_ChessboardDetector(const cv::Size pattern_size, const int blur_ksize) :
	// 保存预期内角点数量和预处理卷积核尺寸。
	m_pattern_size(pattern_size), m_blur_ksize(blur_ksize)
{

}

JSLAB_ChessboardDetector::~JSLAB_ChessboardDetector()
{

}

bool JSLAB_ChessboardDetector::detect(const cv::Mat& img, std::vector<cv::Point2f>& corners, cv::Mat& draw_corners)
{
	// 清除上一张图产生的角点坐标。
	corners.clear();
	// 保留原图副本，用于显示检测结果。
	img.copyTo(draw_corners);

	// 平滑噪声，使棋盘格边缘形成连续的角点结构。
	cv::Mat blur_img;
	cv::GaussianBlur(img, blur_img, cv::Size(m_blur_ksize, m_blur_ksize), 0, 0);

	// 以整数像素精度检测有序的棋盘格内角点阵。
	if (!cv::findChessboardCorners(blur_img, m_pattern_size, corners))
	{
		return false;
	}

	// 在 5x5 邻域中将每个角点精化到小数像素坐标。
	cv::cornerSubPix(blur_img, corners, cv::Size(5, 5), cv::Size(-1, -1),
	                 cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40, 0.01));

	// 将预览图转为彩色，并叠加有序角点序列。
	cv::cvtColor(draw_corners, draw_corners, cv::COLOR_GRAY2BGR);
	cv::drawChessboardCorners(draw_corners, m_pattern_size, corners, true);
	return true;
}

void generateChessboard3DPoints(const cv::Size pattern_size, const float square_size,
                                 std::vector<cv::Point3f>& object_points)
{
	// 以左上角内角点作为平面世界坐标系原点。
	for (int i = 0; i < pattern_size.height; i++)
	{
		for (int j = 0; j < pattern_size.width; j++)
		{
			// 一个方格边长改变 X 或 Y，全部角点保持在 Z = 0 平面。
			object_points.push_back(cv::Point3f(j * square_size, i * square_size, 0));
		}
	}
}
