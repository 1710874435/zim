#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <string>

namespace jslab_phase_preview_detail {

// The same physical range is used for both cameras; invalid samples are gray.
inline cv::Mat colorMap(const cv::Mat& values, const cv::Mat& valid,
                        double maximum)
{
    cv::Mat gray, color;
    values.convertTo(gray, CV_8U, 255.0 / maximum);
    cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(96, 96, 96), valid == 0);
    return color;
}

inline cv::Mat panel(const cv::Mat& color, const std::string& title,
                     const std::string& range)
{
    const int width = 720;
    const int height = 400;
    cv::Mat result(height + 76, width, CV_8UC3, cv::Scalar(30, 30, 30));
    const double scale = (std::min)(double(width) / color.cols,
                                    double(height) / color.rows);
    cv::Mat resized;
    cv::resize(color, resized, cv::Size(), scale, scale, cv::INTER_NEAREST);
    resized.copyTo(result(cv::Rect((width - resized.cols) / 2, 32,
                                  resized.cols, resized.rows)));
    cv::putText(result, title, cv::Point(12, 23), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::Mat ramp(12, width - 24, CV_8U);
    for (int x = 0; x < ramp.cols; ++x)
        ramp.col(x).setTo(255.0 * x / (ramp.cols - 1));
    cv::Mat bar;
    cv::applyColorMap(ramp, bar, cv::COLORMAP_TURBO);
    bar.copyTo(result(cv::Rect(12, height + 38, bar.cols, bar.rows)));
    cv::putText(result, range, cv::Point(12, height + 69),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(255, 255, 255),
                1, cv::LINE_AA);
    return result;
}

inline void saveCamera(const std::string& directory, const cv::Mat& modulation,
                       const cv::Mat& phase)
{
    const std::string separator = !directory.empty()
        && directory.back() != '/' && directory.back() != '\\' ? "/" : "";
    const std::string output_dir = directory + separator;
    if (!cv::utils::fs::createDirectories(output_dir))
        CV_Error(cv::Error::StsError, "Cannot create preview directory: " + output_dir);
    if (!cv::imwrite(output_dir + "modulation.png", modulation)
        || !cv::imwrite(output_dir + "absolute_phase.png", phase))
        CV_Error(cv::Error::StsError, "Cannot save preview images: " + output_dir);
    std::cout << "[Phase preview] Saved: " << output_dir << std::endl;
}

} // namespace jslab_phase_preview_detail

// Call after decoding both cameras and before passing phase maps to reconstruction.
// Modulation is B/A for frequency 0. Absolute phase is in radians.
// PNGs preserve the input pixel dimensions and use TURBO colors:
// modulation [0,1], phase [0,2*pi*reference_frequency]; out-of-range colors clip.
// Orders < 0 identify HDR-invalid phase pixels. Modulation is shown independently.
inline void previewDecodedStereoPhase(
    const cv::Mat& left_modulation, const cv::Mat& right_modulation,
    const cv::Mat& left_phase, const cv::Mat& right_phase,
    const cv::Mat& left_order, const cv::Mat& right_order,
    const std::string& left_output_dir, const std::string& right_output_dir,
    double reference_frequency)
{
    const cv::Mat mods[] = {left_modulation, right_modulation};
    const cv::Mat phases[] = {left_phase, right_phase};
    const cv::Mat orders[] = {left_order, right_order};
    const std::string dirs[] = {left_output_dir, right_output_dir};
    const std::string names[] = {"Left", "Right"};
    CV_Assert(std::isfinite(reference_frequency) && reference_frequency > 0);
    const double phase_max = 2.0 * CV_PI * reference_frequency;
    cv::Mat rows[2];
    for (int i = 0; i < 2; ++i) {
        CV_Assert(!phases[i].empty() && phases[i].type() == CV_64FC1
            && mods[i].type() == CV_64FC1 && orders[i].type() == CV_32SC1
            && mods[i].size() == phases[i].size()
            && orders[i].size() == phases[i].size());
        cv::Mat mod_valid, phase_valid;
        cv::inRange(mods[i], 0.0, DBL_MAX, mod_valid);
        cv::inRange(phases[i], -DBL_MAX, DBL_MAX, phase_valid);
        cv::bitwise_and(phase_valid, orders[i] >= 0, phase_valid);
        const cv::Mat mod_color = jslab_phase_preview_detail::colorMap(
            mods[i], mod_valid, 1.0);
        const cv::Mat phase_color = jslab_phase_preview_detail::colorMap(
            phases[i], phase_valid, phase_max);
        jslab_phase_preview_detail::saveCamera(dirs[i], mod_color, phase_color);
        const cv::Mat panels[] = {
            jslab_phase_preview_detail::panel(mod_color, names[i] + " modulation B/A (frequency 0)",
                "0                                      1 (colors clip outside range)"),
            jslab_phase_preview_detail::panel(phase_color, names[i] + " absolute phase (rad)",
                "0 to " + cv::format("%.3f", phase_max) + " rad; gray = invalid order/sample")
        };
        cv::hconcat(panels, 2, rows[i]);
    }
    cv::Mat canvas;
    cv::vconcat(rows, 2, canvas);
    const std::string window = "Decoded stereo phase - press any key to reconstruct";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, canvas.cols, canvas.rows);
    cv::imshow(window, canvas);
    std::cout << "[Phase preview] Focus the image window and press any key to reconstruct."
              << std::endl;
    cv::waitKey(0);
    cv::destroyWindow(window);
}
