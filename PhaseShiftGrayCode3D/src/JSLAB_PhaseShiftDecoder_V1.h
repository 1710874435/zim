#pragma once

#include "JSLAB_PhaseShiftDecoder.h"
#include <opencv2/core.hpp>
#include <array>

/**
  * @brief V1 multi-frequency phase-shift decoder implementation
  */
class JSLAB_PhaseShiftDecoderV1 : public JSLAB_PhaseShiftDecoder
{
public:
	JSLAB_PhaseShiftDecoderV1(const Config& config);
	~JSLAB_PhaseShiftDecoderV1() override;

	void setImages(const std::vector<cv::Mat>& images) override final;

	void startCalculate(cv::Mat& phase) override final;

	/**
	  * @brief Not supported in V1 returns an empty Mat
	  */
	void getOrderMap(cv::Mat& k_map) override final;

	/**
	  * @brief Not supported in V1 returns an empty Mat
	  */
	void getModulationMap(cv::Mat& mod_map) override final;

private:
	std::vector<cv::Mat> m_images;
	class JSLAB_MultiFreqPhaseShiftDecodeBodyV1* m_body;
};

/**
  * @brief V1 parallel processing body for multi-frequency phase-shift decoding
  */
class JSLAB_MultiFreqPhaseShiftDecodeBodyV1 : public cv::ParallelLoopBody
{
public:
	JSLAB_MultiFreqPhaseShiftDecodeBodyV1(const JSLAB_PhaseShiftDecoder::Config& config);

	~JSLAB_MultiFreqPhaseShiftDecodeBodyV1();

	void operator()(const cv::Range& range) const override final;

	void getPhase(cv::Mat& phase) const;

	void setImages(const std::vector<cv::Mat>& images);

private:
	std::vector<std::vector<cv::Mat>> m_images;
	mutable cv::Mat m_phase;
	const JSLAB_PhaseShiftDecoder::Config m_config;   
	bool m_init;
	std::array<std::vector<double>, 3> m_sin_tmp;
	std::array<std::vector<double>, 3> m_cos_tmp;
};
