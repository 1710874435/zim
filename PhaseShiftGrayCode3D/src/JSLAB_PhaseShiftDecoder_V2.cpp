#include "JSLAB_PhaseShiftDecoder_V2.h"
#include <iostream>

const double PI = 3.14159265358979323846;

class JSLAB_ImageTypeConvertBodyV2 : public cv::ParallelLoopBody
{
public:
	JSLAB_ImageTypeConvertBodyV2(const std::vector<cv::Mat>& images,
	                              std::vector<cv::Mat>& images_double,
	                              double sigma) :
		m_images(images),
		m_images_double(images_double),
		m_sigma(sigma)
	{

	}
	void operator()(const cv::Range& range) const override final
	{
		for (int i = range.start; i < range.end; ++i)
		{
			m_images[i].convertTo(m_images_double[i], CV_64FC1);
			// Pre-blur: sigma=0 lets OpenCV derive sigma from the 3x3 kernel (lighter);
			// sigma>0 applies explicit smoothing strength.
			cv::GaussianBlur(m_images_double[i], m_images_double[i],
			                 cv::Size(3, 3), m_sigma, m_sigma, cv::BORDER_REPLICATE);
		}
	}

private:
	const std::vector<cv::Mat>& m_images;
	std::vector<cv::Mat>& m_images_double;
	double m_sigma;
};

// JSLAB_MultiFreqPhaseShiftDecodeBodyV2 implementation

JSLAB_MultiFreqPhaseShiftDecodeBodyV2::JSLAB_MultiFreqPhaseShiftDecodeBodyV2(const JSLAB_PhaseShiftDecoder::Config& config) :
	m_config(config),
	m_init(false)
{
    if (m_config.enable_hdr) {
        CV_Assert(m_config.hdr_saturation_gray > 0 && m_config.hdr_min_amplitude > 0);
        for (int i = 0; i < 3; ++i)
            m_hdr_fit[i].reset(new JSLAB_HDRPhaseFit(m_config.shift_num[i]));
    }
}

JSLAB_MultiFreqPhaseShiftDecodeBodyV2::~JSLAB_MultiFreqPhaseShiftDecodeBodyV2()
{

}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV2::operator()(const cv::Range& range) const
{
	for (int r = range.start; r != range.end; r++)
	{
		double* phase_ptr = m_phase.ptr<double>(r);
		int*    k_ptr     = m_k_map.ptr<int>(r);
		double* mod_ptr   = m_modulation_map.ptr<double>(r);

		for (int jcol = 0; jcol < m_phase.cols; jcol++)
		{
			// Compute wrapped phase for each of the three frequencies
			double wrap_phase[3];
			int i = 0;
			for (; i < 3; i++)
			{
				const int inum = m_config.shift_num[i];
				double sin_sum  = 0;
				double cos_sum  = 0;
				double sum_gray = 0;
                if (m_config.enable_hdr) {
                    double intensities[12];
                    for (int k = 0; k < inum; ++k) {
                        const cv::Mat& frame = m_images[i][k];
                        intensities[k] = frame.depth() == CV_8U
                            ? frame.ptr<uchar>(r)[jcol] : frame.ptr<double>(r)[jcol];
                    }
                    cv::Vec3d beta;
                    if (!m_hdr_fit[i]->solve(intensities, m_config.hdr_saturation_gray, beta)
                        || std::hypot(beta[1], beta[2]) <= m_config.hdr_min_amplitude
                        || beta[0] <= 1e-3) {
                        i = -1;
                        break;
                    }
                    cos_sum = beta[1] * inum / 2.0;
                    sin_sum = beta[2] * inum / 2.0;
                    sum_gray = beta[0] * inum;
                } else for (int k = 0; k < inum; k++)
				{
					const double sin_tmp   = m_sin_tmp[i][k];
					const double cos_tmp   = m_cos_tmp[i][k];
					const double phase_tmp = m_images[i][k].at<double>(r, jcol);
					sin_sum  -= sin_tmp * phase_tmp;
					cos_sum  += cos_tmp * phase_tmp;
					sum_gray += phase_tmp;
				}
				wrap_phase[i] = atan2(sin_sum, cos_sum);

				// Compute modulation amplitude and apply threshold mask
				double b        = sin_sum * sin_sum + cos_sum * cos_sum;
				double mean_val = sum_gray / inum;
				b               = 2.0f * sqrt(b) / m_config.shift_num[i];

				// Compute normalized modulation and store it for frequency 0
				double norm_mod = 0.0;
				if (mean_val > 1e-3)
					norm_mod = b / mean_val;
				if (i == 0)
					mod_ptr[jcol] = norm_mod;

				if (norm_mod < m_config.modulation_threshold)
				{
					i = -1;
					break;
				}
			}

			if (i == -1)
				continue;

			// Phase unwrapping via multi-frequency heterodyne
			double phase12, phase23, phase123;

			if (wrap_phase[0] > wrap_phase[1])
				phase12 = wrap_phase[0] - wrap_phase[1];
			else
				phase12 = wrap_phase[0] - wrap_phase[1] + 2 * PI;

			if (m_config.freq[1] > m_config.freq[2])
			{
				if (wrap_phase[1] > wrap_phase[2])
					phase23 = wrap_phase[1] - wrap_phase[2];
				else
					phase23 = wrap_phase[1] - wrap_phase[2] + 2 * PI;
			}
			else
			{
				if (wrap_phase[2] > wrap_phase[1])
					phase23 = wrap_phase[2] - wrap_phase[1];
				else
					phase23 = wrap_phase[2] - wrap_phase[1] + 2 * PI;
			}

			if (phase12 > phase23)
				phase123 = phase12 - phase23;
			else
				phase123 = phase12 - phase23 + 2 * PI;

			int f12 = std::abs(static_cast<int>(
				m_config.freq[0] - m_config.freq[1]));

			int f23 = std::abs(static_cast<int>(
				m_config.freq[1] - m_config.freq[2]));

			double uphase12 = phase12 + round((phase123 * f12 - phase12) / (2 * PI)) * 2 * PI;

			double uphase1  = wrap_phase[0] + round((uphase12 * m_config.freq[0] / f12 - wrap_phase[0]) / (2 * PI)) * 2 * PI;
			double k_val    = round((uphase12 * (double)m_config.freq[0] / f12 - wrap_phase[0]) / (2 * PI));

			double uphase23 = phase23 + round((phase123 * f23 - phase23) / (2 * PI)) * 2 * PI;

			double uphase2  = wrap_phase[1] + round((uphase23 * m_config.freq[1] / f23 - wrap_phase[1]) / (2 * PI)) * 2 * PI;
			uphase2 = (uphase2 * m_config.freq[0]) / m_config.freq[1];

			double uphase3  = wrap_phase[2] + round((uphase23 * m_config.freq[2] / f23 - wrap_phase[2]) / (2 * PI)) * 2 * PI;
			uphase3 = (uphase3 * m_config.freq[0]) / m_config.freq[2];

			// Select final phase output:
			//   avg_three_phase=true  -> average of all three (more redundancy, smoother)
			//   avg_three_phase=false -> uphase1 only (default, faster)
			if (m_config.avg_three_phase)
				phase_ptr[jcol] = (uphase1 + uphase2 + uphase3) / 3.0;
			else
				phase_ptr[jcol] = uphase1;

			k_ptr[jcol] = (int)k_val;
            m_valid.ptr<uchar>(r)[jcol] = 255;
		}
	}
}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV2::getPhase(cv::Mat& phase) const
{
	phase = m_phase.clone();
}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV2::getOrderMap(cv::Mat& k_map) const
{
	k_map = m_k_map.clone();
}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV2::getModulationMap(cv::Mat& mod_map) const
{
	mod_map = m_modulation_map.clone();
}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV2::setImages(const std::vector<cv::Mat>& images)
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

        m_valid = cv::Mat::zeros(images[0].size(), CV_8UC1);
		m_phase          = cv::Mat::zeros(images[0].size(), CV_64FC1);
		m_k_map          = cv::Mat::zeros(images[0].size(), CV_32SC1);
		m_modulation_map = cv::Mat::zeros(images[0].size(), CV_64FC1);
		m_images.resize(3);
		m_images[0].resize(m_config.shift_num[0]);
		m_images[1].resize(m_config.shift_num[1]);
		m_images[2].resize(m_config.shift_num[2]);

		m_init = true;
	}

    CV_Assert(images[0].size() == m_phase.size());
    m_valid = 0;
	m_phase          = 0;
	m_k_map          = m_config.enable_hdr ? -1 : 0;
	m_modulation_map = 0;
	std::copy(std::begin(images),
	          std::begin(images) + m_config.shift_num[0],
	          std::begin(m_images[0]));
	std::copy(std::begin(images) + m_config.shift_num[0],
	          std::begin(images) + m_config.shift_num[0] + m_config.shift_num[1],
	          std::begin(m_images[1]));
	std::copy(std::begin(images) + m_config.shift_num[0] + m_config.shift_num[1],
	          std::begin(images) + m_config.shift_num[0] + m_config.shift_num[1] + m_config.shift_num[2],
	          std::begin(m_images[2]));
}


JSLAB_PhaseShiftDecoderV2::JSLAB_PhaseShiftDecoderV2(const Config& config) :
	m_config(config)
{
	m_body = new JSLAB_MultiFreqPhaseShiftDecodeBodyV2(config);
	m_images.resize(config.shift_num[0] + config.shift_num[1] + config.shift_num[2]);
}

JSLAB_PhaseShiftDecoderV2::~JSLAB_PhaseShiftDecoderV2()
{
	std::cout << "[MEMLEAK_CHECK] ~JSLAB_PhaseShiftDecoderV2 called, deleting m_body" << std::endl;
	delete m_body;
	m_body = nullptr;
}

void JSLAB_PhaseShiftDecoderV2::setImages(const std::vector<cv::Mat>& images)
{
    CV_Assert(images.size() == m_images.size());
    for (const cv::Mat& image : images)
        CV_Assert(!image.empty() && image.size() == images[0].size()
            && (image.type() == CV_8UC1 || image.type() == CV_64FC1));
    if (m_config.enable_hdr) {
        // Classify saturation on RAW samples, before any spatial blur.
        // Retaining 8-bit frames also avoids 24 full-size double buffers.
        m_images = images;
    } else {
	cv::parallel_for_(cv::Range(0, images.size()),
	                  JSLAB_ImageTypeConvertBodyV2(images, m_images, m_config.preblur_sigma));
    }

	m_body->setImages(m_images);
}

void JSLAB_PhaseShiftDecoderV2::startCalculate(cv::Mat& phase)
{
	cv::parallel_for_(cv::Range(0, m_images[0].rows), *m_body);

	m_body->getPhase(phase);

	// Optional post-decode smoothing to reduce pixel-level phase noise
    if (m_config.post_smooth) {
        if (m_config.enable_hdr) {
            cv::Mat valid, weight;
            m_body->getValidMask(valid);
            valid.convertTo(weight, CV_64F, 1.0 / 255.0);
            cv::GaussianBlur(phase, phase, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
            cv::GaussianBlur(weight, weight, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
            cv::max(weight, 1e-12, weight);
            cv::divide(phase, weight, phase);
            phase.setTo(0, valid == 0);
        } else {
            cv::GaussianBlur(phase, phase, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
        }
    }
}

void JSLAB_MultiFreqPhaseShiftDecodeBodyV2::getValidMask(cv::Mat& valid) const
{
    valid = m_valid.clone();
}

void JSLAB_PhaseShiftDecoderV2::getOrderMap(cv::Mat& k_map)
{
	m_body->getOrderMap(k_map);
}

void JSLAB_PhaseShiftDecoderV2::getModulationMap(cv::Mat& mod_map)
{
	m_body->getModulationMap(mod_map);
}
