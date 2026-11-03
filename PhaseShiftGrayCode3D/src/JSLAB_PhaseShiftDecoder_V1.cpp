#include "JSLAB_PhaseShiftDecoder_V1.h"
#include <iostream>

const double PI = 3.14159265358979323846;

class JSLAB_ImageTypeConvertBody : public cv::ParallelLoopBody
{
public:
	JSLAB_ImageTypeConvertBody(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& images_double) :
		m_images(images),
		m_images_double(images_double)
	{

	}
	void operator()(const cv::Range& range) const override final
	{
		for (int i = range.start; i < range.end; ++i)
		{
			m_images[i].convertTo(m_images_double[i], CV_64FC1);
		}
	}


private:
	const std::vector<cv::Mat>& m_images;
	std::vector<cv::Mat>& m_images_double;
};

// JSLAB_MultiFreqPhaseShiftDecodeBodyV1 implementation

JSLAB_MultiFreqPhaseShiftDecodeBodyV1::JSLAB_MultiFreqPhaseShiftDecodeBodyV1(const JSLAB_PhaseShiftDecoder::Config& config) :
	m_config(config),
	m_init(false)
{

}

JSLAB_MultiFreqPhaseShiftDecodeBodyV1::~JSLAB_MultiFreqPhaseShiftDecodeBodyV1()
{

}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV1::operator()(const cv::Range& range) const
{
	for (int r = range.start; r != range.end; r++)
	{
		double* phase_ptr = m_phase.ptr<double>(r);

		for (int jcol = 0; jcol < m_phase.cols; jcol++)
		{
			// Compute wrapped phase for each of the three frequencies
			double wrap_phase[3];
			int i = 0;
			for (; i < 3; i++)
			{
				const int inum = m_config.shift_num[i];
				double sin_sum = 0;
				double cos_sum = 0;
				for (int k = 0; k < inum; k++)
				{
					const double sin_tmp = m_sin_tmp[i][k];
					const double cos_tmp = m_cos_tmp[i][k];
					const double phase_tmp = m_images[i][k].at<double>(r, jcol);
					sin_sum -= sin_tmp * phase_tmp;
					cos_sum += cos_tmp * phase_tmp;
				}
				wrap_phase[i] = atan2(sin_sum, cos_sum);

				// Compute phase modulation and apply threshold mask
				double b = sin_sum * sin_sum + cos_sum * cos_sum;
				b = 2.0f * sqrt(b) / m_config.shift_num[i];

				if (b < m_config.modulation_threshold)
				{
					i = -1;
					break;
				}
			}

			if (i == -1)
			{
				continue;
			}

			// Phase unwrapping via multi-frequency heterodyne
			double phase12, phase23, phase123;
			if (wrap_phase[0] > wrap_phase[1])
			{
				phase12 = wrap_phase[0] - wrap_phase[1];
			}
			else
			{
				phase12 = wrap_phase[0] - wrap_phase[1] + 2 * PI;
			}

			if (wrap_phase[1] > wrap_phase[2])
			{
				phase23 = wrap_phase[1] - wrap_phase[2];
			}
			else
			{
				phase23 = wrap_phase[1] - wrap_phase[2] + 2 * PI;
			}

			if (phase12 > phase23)
			{
				phase123 = phase12 - phase23;
			}
			else
			{
				phase123 = phase12 - phase23 + 2 * PI;
			}

			int f12 = m_config.freq[0] - m_config.freq[1];
			int f23 = m_config.freq[1] - m_config.freq[2];

			double uphase12;
			uphase12 = phase12 + round((phase123 * f12 - phase12) / (2 * PI)) * 2 * PI;

			double uphase1;
			uphase1 = wrap_phase[0] + round((uphase12 * m_config.freq[0] / f12 - wrap_phase[0]) / (2 * PI)) * 2 * PI;

			double uphase23;
			uphase23 = phase23 + round((phase123 * f23 - phase23) / (2 * PI)) * 2 * PI;

			double uphase2;
			uphase2 = wrap_phase[1] + round((uphase23 * m_config.freq[1] / f23 - wrap_phase[1]) / (2 * PI)) * 2 * PI;
			uphase2 = (uphase2 * m_config.freq[0]) / m_config.freq[1];

			double uphase3;
			uphase3 = wrap_phase[2] + round((uphase23 * m_config.freq[2] / f23 - wrap_phase[2]) / (2 * PI)) * 2 * PI;
			uphase3 = (uphase3 * m_config.freq[0]) / m_config.freq[2];


			if (abs(uphase1 - uphase2) > 1.0)
			{
				continue;
			}

			phase_ptr[jcol] = (uphase1 + uphase2 + uphase3) / 3.0;
		}
	}

}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV1::getPhase(cv::Mat& phase) const
{
	phase = m_phase.clone();
}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV1::setImages(const std::vector<cv::Mat>& images)
{
	if (!m_init)
	{
		for (int i = 0; i < 3; i++)
		{
			const int inum = m_config.shift_num[i];
			for (int k = 0; k < inum; k++)
			{
				m_sin_tmp[i].push_back(sin(2.0 * PI * k / inum));
				m_cos_tmp[i].push_back(cos(2.0 * PI * k / inum));
			}
		}

		m_phase = cv::Mat::zeros(images[0].size(), CV_64FC1);

		m_images.resize(3);
		m_images[0].resize(m_config.shift_num[0]);
		m_images[1].resize(m_config.shift_num[1]);
		m_images[2].resize(m_config.shift_num[2]);

		m_init = true;
	}

	m_phase = 0;

	std::copy(std::begin(images), std::begin(images) + m_config.shift_num[0], std::begin(m_images[0]));
	std::copy(std::begin(images) + m_config.shift_num[0], std::begin(images) + m_config.shift_num[0] + m_config.shift_num[1], std::begin(m_images[1]));
	std::copy(std::begin(images) + m_config.shift_num[0] + m_config.shift_num[1], std::begin(images) + m_config.shift_num[0] + m_config.shift_num[1] + m_config.shift_num[2], std::begin(m_images[2]));
}

JSLAB_PhaseShiftDecoderV1::JSLAB_PhaseShiftDecoderV1(const Config& config)
{
	m_body = new JSLAB_MultiFreqPhaseShiftDecodeBodyV1(config);
	m_images.resize(config.shift_num[0] + config.shift_num[1] + config.shift_num[2]);
}

JSLAB_PhaseShiftDecoderV1::~JSLAB_PhaseShiftDecoderV1()
{
	std::cout << "[MEMLEAK_CHECK] ~JSLAB_PhaseShiftDecoderV1 called, deleting m_body" << std::endl;
	delete m_body;
	m_body = nullptr;
}

void JSLAB_PhaseShiftDecoderV1::setImages(const std::vector<cv::Mat>& images)
{
	cv::parallel_for_(cv::Range(0, images.size()), JSLAB_ImageTypeConvertBody(images, m_images));

	m_body->setImages(m_images);
}

void JSLAB_PhaseShiftDecoderV1::startCalculate(cv::Mat& phase)
{
	cv::parallel_for_(cv::Range(0, m_images[0].rows), *m_body);

	m_body->getPhase(phase);
}

void JSLAB_PhaseShiftDecoderV1::getOrderMap(cv::Mat& k_map)
{
	// V1 does not support order map
	k_map = cv::Mat();
}

void JSLAB_PhaseShiftDecoderV1::getModulationMap(cv::Mat& mod_map)
{
	// V1 does not support modulation map
	mod_map = cv::Mat();
}
