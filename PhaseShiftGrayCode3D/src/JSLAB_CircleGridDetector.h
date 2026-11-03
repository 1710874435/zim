#pragma once
#include <opencv2/opencv.hpp>
#include "dllexport.h"

/**
  * @brief Circular-dot target detector
  */
class DLL_EXPORT JSLAB_CircleGridDetector
{
public:

	/**
	  * @brief Constructor
	  * @param [in] pattern_size         Number of dots (cols x rows)
	  * @param [in] blur_ksize           Gaussian blur kernel size applied before detection
	  * @param [in] maxArea              Maximum blob area in pixels
	  * @param [in] minArea              Minimum blob area in pixels
	  * @param [in] filterByArea         Enable area-based blob filtering
	  * @param [in] filterByColor        Enable color-based blob filtering
	  * @param [in] blobColor            Target blob intensity (0 = dark blobs, 255 = bright blobs)
	  * @param [in] minDistBetweenBlobs  Minimum distance between blob centers
	  * @param [in] minCircularity       Minimum circularity score [0, 1]
	  * @param [in] maxCircularity       Maximum circularity score [0, 1]
	  * @param [in] filterByCircularity  Enable circularity-based blob filtering
	  */
	JSLAB_CircleGridDetector(const cv::Size pattern_size, const int blur_ksize,
		const double maxArea = 100000, const double minArea = 1000,
		const bool filterByArea = true, const bool filterByColor = true,
		const unsigned char blobColor = 255, const float minDistBetweenBlobs = 30,
		const float minCircularity = 0.6f, const float maxCircularity = 1.0f,
		const bool filterByCircularity = true);

	/**
	  * @brief Destructor
	  */
	~JSLAB_CircleGridDetector();

	/**
	  * @brief Detect the circular-dot grid in a grayscale image
	  * @param [in]  img          Input grayscale image
	  * @param [out] corners      Detected dot center positions
	  * @param [out] draw_corners Color image with dot centers drawn on it
	  * @return true if detection succeeded, false otherwise
	  */
	bool detect(const cv::Mat& img, std::vector<cv::Point2f>& corners, cv::Mat& draw_corners);

private:
	const cv::Size m_pattern_size;
	const int m_blur_ksize;
	cv::Ptr<cv::FeatureDetector> m_blobDetector;
};

/**
  * @brief Generate 3-D world coordinates for circular-dot target points
  * @param [in]  pattern_size  Number of dots (cols x rows)
  * @param [in]  square_size   Physical dot pitch (center-to-center distance)
  * @param [out] object_points Output 3-D point list (Z = 0 plane)
  */
DLL_EXPORT void generateCircleGrid3DPoints(const cv::Size pattern_size, const float square_size,
                                            std::vector<cv::Point3f>& object_points);
