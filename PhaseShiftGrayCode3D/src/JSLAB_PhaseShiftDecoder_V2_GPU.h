#pragma once

#include "JSLAB_PhaseShiftDecoder.h"
#include <opencv2/core.hpp>
#include <vector>

/**
  * @brief GPU (CUDA) port of the V2 multi-frequency phase-shift decoder.
  *
  * Mirrors JSLAB_PhaseShiftDecoderV2 semantics (3-frequency heterodyne
  * unwrapping, order map, modulation map) but performs the per-pixel decode on
  * the device.  The public interface stays cv::Mat for drop-in compatibility;
  * all heavy compute runs on the GPU with a single host-to-device upload of the
  * raw frames and a single device-to-host download of the results.
  *
 * Precision: raw frames are uploaded as float (exact for 8/16-bit inputs), but
 * the phase math (pre-blur, atan2, heterodyne unwrap round()) runs in double so
 * the unwrap order decisions stay bit-identical to the CPU decoder.
 */
class DLL_EXPORT JSLAB_PhaseShiftDecoderV2_GPU : public JSLAB_PhaseShiftDecoder
{
public:
	JSLAB_PhaseShiftDecoderV2_GPU(const Config& config);
	~JSLAB_PhaseShiftDecoderV2_GPU() override;

	void setImages(const std::vector<cv::Mat>& images) override final;

	void startCalculate(cv::Mat& phase) override final;

	void getOrderMap(cv::Mat& k_map) override final;

	void getModulationMap(cv::Mat& mod_map) override final;

private:
	const Config m_config;            // stored by value so lifetime is guaranteed
	std::vector<cv::Mat> m_images;    // raw input frames (not pre-converted)
	cv::Mat m_phase;                  // CV_64FC1 decoded absolute phase
	cv::Mat m_k_map;                  // CV_32SC1 fringe order map
	cv::Mat m_modulation_map;         // CV_64FC1 normalized modulation map
};

// ---------------------------------------------------------------------------
// CUDA host launcher (implemented in JSLAB_PhaseShiftDecoder_V2_GPU.cu).
// Uploads the raw frames once, runs the 3x3 pre-blur + decode kernels on the
// device, and downloads phase / order / modulation maps.
//   phase   : CV_64FC1
//   k_map   : CV_32SC1
//   mod_map : CV_64FC1
// ---------------------------------------------------------------------------
void JSLAB_DecodePhase_GPU(const std::vector<cv::Mat>& images,
                           const JSLAB_PhaseShiftDecoder::Config& cfg,
                           cv::Mat& phase, cv::Mat& k_map, cv::Mat& mod_map);

// Initializes the CUDA context up front so its one-time setup is not charged to
// the first decode.  Optional; call once before timing / first use.
DLL_EXPORT void JSLAB_DecodePhase_GPU_Warmup();

// Frees the persistent device buffers used by JSLAB_DecodePhase_GPU.  Optional;
// otherwise they are reclaimed at process exit.
void JSLAB_DecodePhase_GPU_Release();
