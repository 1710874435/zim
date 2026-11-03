#pragma once

#include <opencv2/core.hpp>
#include <string>
#include "JSLAB_PhaseShiftDecoder.h"

static bool isValidPoint(const cv::Vec3f& point);

// JSLAB_PhaseDisparityExporter.cpp
// Decode the global (absolute) phase maps of one phase-shift group, run the
// stereo reconstruction, and export phase maps + disparity map into out_dir.
bool exportPhaseAndDisparityMaps(const std::string& pathL,
                                 const std::string& pathR,
                                 const std::string& calib_xml_path,
                                 const std::string& out_dir,
                                 const JSLAB_PhaseShiftDecoder::Config& config);
