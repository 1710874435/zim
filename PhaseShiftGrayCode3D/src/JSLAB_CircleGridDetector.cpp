#include "JSLAB_CircleGridDetector.h"

JSLAB_CircleGridDetector::JSLAB_CircleGridDetector(const cv::Size pattern_size, const int blur_ksize,
	const double maxArea, const double minArea,
	const bool filterByArea, const bool filterByColor,
	const unsigned char blobColor, const float minDistBetweenBlobs,
	const float minCircularity, const float maxCircularity,
	const bool filterByCircularity) :
	// 保存标靶点阵布局和每次检测使用的模糊参数。
	m_pattern_size(pattern_size), m_blur_ksize(blur_ksize)
{
	// 配置哪些图像区域可以作为标定圆点候选。
	cv::SimpleBlobDetector::Params params;
	params.maxArea              = maxArea;              // 圆点候选允许的最大像素面积。
	params.minArea              = minArea;              // 圆点候选允许的最小像素面积。
	params.filterByArea         = filterByArea;         // 为 true 时启用面积范围筛选。
	params.filterByColor        = filterByColor;        // 为 true 时启用中心亮度筛选。
	params.blobColor            = blobColor;            // 0 表示黑色圆点，255 表示白色圆点。
	params.minDistBetweenBlobs  = minDistBetweenBlobs;   // 限制相邻圆心的最小距离。
	params.minCircularity       = minCircularity;        // 圆形度 4*pi*面积/(周长*周长) 的下限。
	params.maxCircularity       = maxCircularity;        // 圆形度的上限。
	params.filterByCircularity  = filterByCircularity;   // 为 true 时启用圆形度范围筛选。

	// 创建可复用的检测器，使每张图遵循相同筛选规则。
	m_blobDetector = cv::SimpleBlobDetector::create(params);
}

JSLAB_CircleGridDetector::~JSLAB_CircleGridDetector()
{

}

bool JSLAB_CircleGridDetector::detect(const cv::Mat& img, std::vector<cv::Point2f>& corners, cv::Mat& draw_corners)
{
	// 清除上一张图的圆心结果，准备写入当前图的结果。
	corners.clear();

	// 平滑传感器噪声，再提升对比度以稳定提取圆点。
	cv::Mat gauss_img;
	cv::GaussianBlur(img, gauss_img, cv::Size(m_blur_ksize, m_blur_ksize), 0.5, 0.5);
	gauss_img.convertTo(gauss_img, -1, 1.5, 0);

	// 检测对称圆点阵，并将有序圆心像素坐标写入 corners。
	bool lret = cv::findCirclesGrid(gauss_img, m_pattern_size, corners,
	                                cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING,
	                                m_blobDetector);
	if (!lret)
	{
		std::cout << "findCirclesGrid failed" << std::endl;
		// 保留原图，供调用者显示检测失败的画面。
		img.copyTo(draw_corners);
		return false;
	}

	// 统一圆心排列方向，使同一物理圆点保持相同数组下标。
	if (!corners.empty())
	{
		cv::Point2f first_pt = corners.front();
		cv::Point2f last_pt = corners.back();

		if (first_pt.y > last_pt.y || (std::abs(first_pt.y - last_pt.y) < 10.0f && first_pt.x > last_pt.x))
		{
			std::reverse(corners.begin(), corners.end());
		}
	}

	// 生成彩色预览图，绘制圆心顺序，并缩小为便于显示的尺寸。
	cv::cvtColor(img, draw_corners, cv::COLOR_GRAY2BGR);
	cv::drawChessboardCorners(draw_corners, m_pattern_size, corners, lret);
	cv::resize(draw_corners, draw_corners, cv::Size(), 0.5, 0.5);

	return true;
}

void generateCircleGrid3DPoints(const cv::Size pattern_size, const float square_size,
                                 std::vector<cv::Point3f>& object_points)
{
	// 以标靶左上角圆点作为世界坐标系原点。
	for (int i = 0; i < pattern_size.height; i++)
	{
		for (int j = 0; j < pattern_size.width; j++)
		{
			// 全部圆点位于标靶平面，因此 Z 坐标为零。
			object_points.push_back(cv::Point3f(j * square_size, i * square_size, 0));
		}
	}
}
