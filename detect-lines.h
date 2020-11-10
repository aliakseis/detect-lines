#pragma once

#include "opencv2/imgproc.hpp"

#include <string>
#include <tuple>
#include <vector>

#include <functional>

std::vector<std::tuple<double, double, double, double, double>> calculating(
        const std::string& filename, std::function<void(const std::string&, cv::InputArray)> do_imshow = {});
