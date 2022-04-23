#pragma once
/// @file

#include "opencv2/imgproc.hpp"

#include <string>
#include <tuple>
#include <vector>

#include <functional>

namespace imaging {

/**
@brief calculates orientation and coherency by using a gradient structure tensor.
In mathematics, the gradient structure tensor (also referred to as the second-moment matrix,
the second order moment tensor, the inertia tensor, etc.) is a matrix derived from the gradient of a function.
It summarizes the predominant directions of the gradient in a specified neighborhood of a point,
and the degree to which those directions are coherent (coherency).
The gradient structure tensor is widely used in image processing and computer vision for 2D/3D image segmentation,
motion detection, adaptive filtration, local image features detection, etc.
Important features of anisotropic images include orientation and coherency of a local anisotropy.
@param w defines a window size
*/
void calcGST(const cv::Mat& inputImg, cv::Mat& imgCoherencyOut, cv::Mat& imgOrientationOut, int w = 52);

/**
@brief An old algorithm using classical computer vision approach.
Tried a couple of approaches using OpenCV so far. The first one basically consisted of the next steps:
Filtering, background substruction -> adaptiveThreshold -> thinning -> HoughLinesP -> and then filtering and merging of lines.
The second approach comprised of search for the beginnings of short stripes with SURF and movement to the left and up along long lines.
The third approach had beei tried tried: doing the Fourier transform for frames - image fragments (a 4-dimensional matrix is obtained), then finding basic patterns using PCA.
Have tried to select lines using adaptiveThreshold using original image, then teach the multilayer perceptron based on this threshold and the PCA result so that it would yield "refined" threshold.
An attempt was made to select the parameters resulting in a cleared threshold for further treatment - it works occasionally, but the result is very unstable.
Unfortulately all the approaches above work only with few selected "good" images.
@param filename defines an imput image file name.
@param do_imshow defines an image displaying and/or serializing callback.
*/
std::vector<std::tuple<double, double, double, double, double>> calculating(
    const std::string& filename, std::function<void(const cv::String&, cv::InputArray)> do_imshow = {});

/**
@brief Applying an anisotropic filter.
@param src defines a source image.
*/
cv::Mat filter(const cv::Mat& src);

/**
@brief Visualizing an image structure.
@param src defines a source image.
*/
std::vector<std::vector<cv::Point>> transmogrify(const cv::Mat& src);

}  // namespace imaging
