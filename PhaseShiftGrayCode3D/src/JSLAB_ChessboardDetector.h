#pragma once
#include <opencv2/opencv.hpp>
#include "dllexport.h"

/**
  * @brief Chessboard corner detector
  */
class DLL_EXPORT JSLAB_ChessboardDetector
{
public:

	/**
	  * @brief Constructor
	  * @param [in] pattern_size Number of inner corners (cols x rows)
	  * @param [in] blur_ksize   Gaussian blur kernel size applied before detection
	  */
	JSLAB_ChessboardDetector(const cv::Size pattern_size, const int blur_ksize);

	/**
	  * @brief Destructor
	  */
	~JSLAB_ChessboardDetector();

	/**
	  * @brief Detect chessboard corners in a grayscale image
	  * @param [in]  img          Input grayscale image
	  * @param [out] corners      Detected corner positions (sub-pixel refined)
	  * @param [out] draw_corners Color image with corners drawn on it
	  * @return true if detection succeeded, false otherwise
	  */
	bool detect(const cv::Mat& img, std::vector<cv::Point2f>& corners, cv::Mat& draw_corners);

private:
	const cv::Size m_pattern_size;
	const int m_blur_ksize;
};

/**
  * @brief Generate 3-D world coordinates for chessboard corners
  * @param [in]  pattern_size Number of inner corners (cols x rows)
  * @param [in]  square_size  Physical size of one chessboard square
  * @param [out] object_points Output 3-D point list (Z = 0 plane)
  */
DLL_EXPORT void generateChessboard3DPoints(const cv::Size pattern_size, const float square_size,
                                            std::vector<cv::Point3f>& object_points);
