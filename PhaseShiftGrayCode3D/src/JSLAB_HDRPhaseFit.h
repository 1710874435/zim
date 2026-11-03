#pragma once

#include <opencv2/core.hpp>
#include <cmath>
#include <vector>

// I(delta) = A + C*cos(delta) - S*sin(delta). Equal-variance least squares.
// Cache each usable-observation subset; the pixel loop only forms a bit mask
// and multiplies the corresponding pseudoinverse by the raw intensities.
class JSLAB_HDRPhaseFit
{
public:
    explicit JSLAB_HDRPhaseFit(int steps) : m_steps(steps)
    {
        CV_Assert(steps >= 3 && steps <= 12);
        m_fits.resize(size_t(1) << steps);
        const double tau = 2.0 * CV_PI;
        for (size_t mask = 0; mask < m_fits.size(); ++mask) {
            std::vector<int> indices;
            for (int k = 0; k < steps; ++k)
                if (mask & (size_t(1) << k)) indices.push_back(k);
            if (indices.size() < 3) continue;
            cv::Mat design(static_cast<int>(indices.size()), 3, CV_64F);
            for (int j = 0; j < design.rows; ++j) {
                const double delta = tau * indices[j] / steps;
                design.at<double>(j, 0) = 1.0;
                design.at<double>(j, 1) = std::cos(delta);
                design.at<double>(j, 2) = -std::sin(delta);
            }
            cv::SVD svd(design, cv::SVD::NO_UV);
            if (svd.w.at<double>(2) <= 1e-10 * svd.w.at<double>(0)) continue;
            cv::Mat inverse;
            cv::invert(design, inverse, cv::DECOMP_SVD);
            Fit& fit = m_fits[mask];
            fit.weights.assign(steps, cv::Vec3d(0, 0, 0));
            for (int j = 0; j < design.rows; ++j)
                fit.weights[indices[j]] = cv::Vec3d(inverse.at<double>(0, j),
                    inverse.at<double>(1, j), inverse.at<double>(2, j));
            fit.valid = true;
        }
    }

    bool solve(const double* intensities, double saturation, cv::Vec3d& beta) const
    {
        size_t mask = 0;
        for (int k = 0; k < m_steps; ++k)
            if (intensities[k] >= 0.0 && intensities[k] < saturation)
                mask |= size_t(1) << k;
        const Fit& fit = m_fits[mask];
        if (!fit.valid) return false;
        beta = cv::Vec3d(0, 0, 0);
        for (int k = 0; k < m_steps; ++k)
            if (mask & (size_t(1) << k)) beta += fit.weights[k] * intensities[k];
        return std::isfinite(beta[0]) && std::isfinite(beta[1]) && std::isfinite(beta[2]);
    }

private:
    struct Fit { bool valid = false; std::vector<cv::Vec3d> weights; };
    int m_steps;
    std::vector<Fit> m_fits;
};
