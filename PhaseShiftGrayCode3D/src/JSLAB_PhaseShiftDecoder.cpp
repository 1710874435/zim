#include "JSLAB_PhaseShiftDecoder.h"
#include "JSLAB_PhaseShiftDecoder_V1.h"
#include "JSLAB_PhaseShiftDecoder_V2.h"

/**
  * @brief Factory method implementation for JSLAB_PhaseShiftDecoder.
  * @details Creates the concrete decoder that matches the requested version.
  */
cv::Ptr<JSLAB_PhaseShiftDecoder> JSLAB_PhaseShiftDecoder::create(const Config& config, Version version)
{
	// All three frequencies must be non-zero for a valid configuration
	if (config.freq[0] != 0 && config.freq[1] != 0 && config.freq[2] != 0)
	{
		switch(version)
		{
		case Version::V1:
			return cv::makePtr<JSLAB_PhaseShiftDecoderV1>(config);

		case Version::V2:
			return cv::makePtr<JSLAB_PhaseShiftDecoderV2>(config);

		default:
			return cv::makePtr<JSLAB_PhaseShiftDecoderV1>(config);
		}
	}

	// Invalid configuration return null
	return nullptr;
}
