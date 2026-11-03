#include "JSLAB_PhaseShiftDecoder_V2_GPU.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

JSLAB_PhaseShiftDecoderV2_GPU::JSLAB_PhaseShiftDecoderV2_GPU(const Config& config) :
	m_config(config)
{
	m_images.resize(config.shift_num[0] + config.shift_num[1] + config.shift_num[2]);
}

JSLAB_PhaseShiftDecoderV2_GPU::~JSLAB_PhaseShiftDecoderV2_GPU()
{
	std::cout << "[MEMLEAK_CHECK] ~JSLAB_PhaseShiftDecoderV2_GPU called" << std::endl;
}

void JSLAB_PhaseShiftDecoderV2_GPU::setImages(const std::vector<cv::Mat>& images)
{
	// Keep raw frames; upload (as 8-bit) and the fused pre-blur happen on the
	// device during startCalculate to avoid redundant host work and extra copies.
	const int total = m_config.shift_num[0] + m_config.shift_num[1] + m_config.shift_num[2];
	m_images.resize(total);
	for (int i = 0; i < total && i < (int)images.size(); ++i)
		m_images[i] = images[i];
}

void JSLAB_PhaseShiftDecoderV2_GPU::startCalculate(cv::Mat& phase)
{
	JSLAB_DecodePhase_GPU(m_images, m_config, m_phase, m_k_map, m_modulation_map);

	// Deep copy: m_phase is reused (same buffer) on the next decode, so a shallow
	// copy would let a later call silently overwrite a previously returned map.
	phase = m_phase.clone();

	// Optional post-decode smoothing to reduce pixel-level phase noise
	if (m_config.post_smooth)
		cv::GaussianBlur(phase, phase, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
}

void JSLAB_PhaseShiftDecoderV2_GPU::getOrderMap(cv::Mat& k_map)
{
	k_map = m_k_map.clone();
}

void JSLAB_PhaseShiftDecoderV2_GPU::getModulationMap(cv::Mat& mod_map)
{
	mod_map = m_modulation_map.clone();
}
