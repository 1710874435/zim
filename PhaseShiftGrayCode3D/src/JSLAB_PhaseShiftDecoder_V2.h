#pragma once

#include "JSLAB_PhaseShiftDecoder.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include "JSLAB_HDRPhaseFit.h"
#include <memory>

/**
  * @brief V2 multi-frequency phase-shift decoder implementation
  *        (supports order map and modulation map)
  */
class JSLAB_PhaseShiftDecoderV2 : public JSLAB_PhaseShiftDecoder
{
public:
	JSLAB_PhaseShiftDecoderV2(const Config& config);
	~JSLAB_PhaseShiftDecoderV2() override;

	void setImages(const std::vector<cv::Mat>& images) override final;

	void startCalculate(cv::Mat& phase) override final;

	/**
	  * @brief Get the fringe order map computed during the last decode
	  */
	void getOrderMap(cv::Mat& k_map) override final;

	/**
	  * @brief Get the normalized modulation map computed during the last decode
	  */
	void getModulationMap(cv::Mat& mod_map) override final;


private:
	const Config m_config;    // stored by value so lifetime is guaranteed
	std::vector<cv::Mat> m_images;
	class JSLAB_MultiFreqPhaseShiftDecodeBodyV2* m_body;
};

/**
  * @brief V2 parallel processing body for multi-frequency phase-shift decoding
  */
class JSLAB_MultiFreqPhaseShiftDecodeBodyV2 : public cv::ParallelLoopBody
{
public:
	JSLAB_MultiFreqPhaseShiftDecodeBodyV2(const JSLAB_PhaseShiftDecoder::Config& config);

	~JSLAB_MultiFreqPhaseShiftDecodeBodyV2();

	void operator()(const cv::Range& range) const override final;

	void getPhase(cv::Mat& phase) const;

	void setImages(const std::vector<cv::Mat>& images);

	void getOrderMap(cv::Mat& k_map) const;

	void getModulationMap(cv::Mat& mod_map) const;
	void getValidMask(cv::Mat& valid) const;


private:
	std::vector<std::vector<cv::Mat>> m_images;
	mutable cv::Mat m_phase;
	mutable cv::Mat m_valid;
	std::array<std::unique_ptr<JSLAB_HDRPhaseFit>, 3> m_hdr_fit;
	mutable cv::Mat m_k_map;
	mutable cv::Mat m_modulation_map;
	const JSLAB_PhaseShiftDecoder::Config m_config;  
	bool m_init;
	std::array<std::vector<double>, 3> m_sin_tmp;
	std::array<std::vector<double>, 3> m_cos_tmp;
};
