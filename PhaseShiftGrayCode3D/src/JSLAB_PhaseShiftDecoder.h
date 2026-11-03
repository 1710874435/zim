
#pragma once

#include <opencv2/core.hpp>
#include "dllexport.h"

/**
  * @brief Multi-frequency phase-shift decoder (abstract factory interface)
  */
class DLL_EXPORT JSLAB_PhaseShiftDecoder
{
public:
	/**
	  * @brief Algorithm version enumeration
	  */
	enum class Version
	{
		V1,  // Version 1
		V2   // Version 2 (supports order map and modulation map)
	};

	struct Config
	{
		//////////////////// Multi-frequency heterodyne parameters ////////////////////
		double freq[3];       // Fringe frequencies for the three levels
		int shift_num[3];     // Number of phase shifts per frequency level
		///////////////////////////////////////////////////////////////////////////////

		// Modulation threshold: pixels below this value are masked out (set to 0)
		double modulation_threshold = 1.0;

		// --- V2-specific parameters (ignored by V1) ---

		// Sigma for per-image Gaussian pre-blur applied before decoding.
		// 0 = sigma is auto-derived from the kernel size (lighter smoothing).
		// 3.0 = stronger smoothing (original V2 default).
		double preblur_sigma = 3.0;

		// If true, the final absolute phase is the average of uphase1, uphase2
		// and uphase3 (same strategy as V1, more robust).
		// If false, only uphase1 is used (faster, slightly less redundancy).
		bool avg_three_phase = true;

		// If true, apply a 3x3 Gaussian blur to the decoded phase map after
		// calculation to reduce pixel-level noise at the cost of edge sharpness.
		bool post_smooth = true;

        // V2 CPU HDR: fit raw observations satisfying 0 <= I < saturation.
        // Reuses the original multi-frequency unwrapping equations.
        bool enable_hdr = false;
        double hdr_saturation_gray = 255.0;
        double hdr_min_amplitude = 2.0;

	};

	/**
	  * @brief Set input images
	  * @param [in] images Input image sequence
	  */
	virtual void setImages(const std::vector<cv::Mat>& images) = 0;

	/**
	  * @brief Run phase decoding
	  * @param [out] phase Output absolute phase map (CV_64FC1)
	  */
	virtual void startCalculate(cv::Mat& phase) = 0;

	/**
	  * @brief Get order map (V2 only; V1 returns an empty Mat)
	  * @param [out] k_map Output fringe order map (CV_32SC1)
	  */
	virtual void getOrderMap(cv::Mat& k_map) = 0;

	/**
	  * @brief Get modulation map (V2 only; V1 returns an empty Mat)
	  * @param [out] mod_map Output normalized modulation map (CV_64FC1)
	  */
	virtual void getModulationMap(cv::Mat& mod_map) = 0;

	/**
	  * @brief Factory method: create a phase-shift decoder instance
	  * @param [in] config Decoder configuration
	  * @param [in] version Algorithm version (default V1)
	  * @return Smart pointer to the created decoder
	  */
	static cv::Ptr<JSLAB_PhaseShiftDecoder> create(const Config& config, Version version = Version::V1);

	/**
	  * @brief Destructor
	  */
	virtual ~JSLAB_PhaseShiftDecoder() = default;
};
