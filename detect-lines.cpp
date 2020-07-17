#include "detect-lines.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/ximgproc.hpp>

#include <numeric>

using namespace cv;


namespace {

Vec4i extendedLine(const Vec4i& line, double d, double max_coeff) {
    const auto length = hypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = std::min(d / length, max_coeff);
    double xd = (line[2] - line[0]) * coeff;
    double yd = (line[3] - line[1]) * coeff;
    return Vec4d(line[0] - xd, line[1] - yd, line[2] + xd, line[3] + yd);
}

std::vector<Point2i> boundingRectangleContour(const Vec4i& line, float d) {
    // finds coordinates of perpendicular lines with length d in both line points
    const auto length = hypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = d / length;

    // dx:  -dy
    // dy:  dx
    double yd = (line[2] - line[0]) * coeff;
    double xd = -(line[3] - line[1]) * coeff;

    return std::vector<Point2i> {
        Point2i(line[0]-xd, line[1]-yd),
        Point2i(line[0]+xd, line[1]+yd),
        Point2i(line[2]+xd, line[3]+yd),
        Point2i(line[2]-xd, line[3]-yd)
    };
}

bool extendedBoundingRectangleLineEquivalence(const Vec4i& l1, const Vec4i& l2, 
    float extensionLength, float extensionLengthMaxFraction,
    float boundingRectangleThickness) {

    Vec4i el1 = extendedLine(l1, extensionLength, extensionLengthMaxFraction);
    Vec4i el2 = extendedLine(l2, extensionLength, extensionLengthMaxFraction);

    // calculate window around extended line
    // at least one point needs to inside extended bounding rectangle of other line,
    std::vector<Point2i> lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
    return
        pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) >= 0 ||
        pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) >= 0 ||

        pointPolygonTest(lineBoundingContour, cv::Point(l2[0], l2[1]), false) >= 0 ||
        pointPolygonTest(lineBoundingContour, cv::Point(l2[2], l2[3]), false) >= 0;
}

Vec4i HandlePointCloud(const std::vector<Point2i>& pointCloud) {
    //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
    // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
    Vec4f lineParams;
    fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);

    // derive the bounding xs of point cloud
    std::vector<Point2i>::const_iterator minYP;
    std::vector<Point2i>::const_iterator maxYP;
    std::tie(minYP, maxYP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const Point2i& p1, const Point2i& p2) { return p1.y < p2.y; });

    // derive y coords of fitted line
    float m = lineParams[0] / lineParams[1];
    int x1 = ((minYP->y - lineParams[3]) * m) + lineParams[2];
    int x2 = ((maxYP->y - lineParams[3]) * m) + lineParams[2];

    return { x1, minYP->y, x2, maxYP->y };
}

std::vector<Vec4i> reduceLines(const std::vector<Vec4i>& linesP,
    float extensionLength, float extensionLengthMaxFraction,
    float boundingRectangleThickness)
{
    // partition via our partitioning function
    std::vector<int> labels;
    int equilavenceClassesCount = cv::partition(linesP, labels, 
        [extensionLength, extensionLengthMaxFraction, boundingRectangleThickness](const Vec4i& l1, const Vec4i& l2) {
        return extendedBoundingRectangleLineEquivalence(
            l1, l2,
            // line extension length 
            extensionLength,
            // line extension length - as fraction of original line width
            extensionLengthMaxFraction,
            // thickness of bounding rectangle around each line
            boundingRectangleThickness);
    });

    std::vector<std::vector<Vec4i>> groups(equilavenceClassesCount);
    for (int i = 0; i < linesP.size(); i++) {
        const Vec4i& detectedLine = linesP[i];
        groups[labels[i]].push_back(detectedLine);
    }

    equilavenceClassesCount = groups.size();

    // build point clouds out of each equivalence classes
    std::vector<std::vector<Point2i>> pointClouds(equilavenceClassesCount);
    for (int i = 0; i < equilavenceClassesCount; ++i) {
        for (auto &detectedLine : groups[i]) {
            pointClouds[i].emplace_back(detectedLine[0], detectedLine[1]);
            pointClouds[i].emplace_back(detectedLine[2], detectedLine[3]);
        }
}
    std::vector<Vec4i> reducedLines = std::accumulate(
        pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{}, [](std::vector<Vec4i> target, const std::vector<Point2i>& pointCloud) {
        target.push_back(HandlePointCloud(pointCloud));
        return target;
    });

    return reducedLines;
}

template <typename T>
void MergeLines(std::vector<Vec4i>& reducedLines, T sortLam) {
    for (int i = reducedLines.size(); --i >= 0;)
    {
        auto& line = reducedLines[i];
        if (hypot(line[2] - line[0], line[3] - line[1]) > 40) {
            continue;
        }

        auto val = sortLam(line);

        double dist;
        std::vector<Vec4i>::iterator it;
        if (i == 0) {
            it = reducedLines.begin() + 1;
            dist = sortLam(*it) - val;
        }
        else if (i == reducedLines.size() - 1) {
            it = reducedLines.begin() + i - 2;
            dist = val - sortLam(*it);
        }
        else {
            const auto dist1 = val - sortLam(reducedLines[i - 1]);
            const auto dist2 = sortLam(reducedLines[i + 1]) - val;
            if (dist1 < dist2) {
                it = reducedLines.begin() + i - 1;
                dist = dist1;
            }
            else {
                it = reducedLines.begin() + i + 1;
                dist = dist2;
            }
        }

        const auto distY = abs((line[1] + line[3]) / 2 - ((*it)[1] + (*it)[3]) / 2)
            - (abs(line[1] - line[3]) + abs((*it)[1] - (*it)[3])) / 2;

        const auto threshold = 2.5;
        const auto thresholdY = 25;
        if (dist > threshold || distY > thresholdY) {
            reducedLines.erase(reducedLines.begin() + i);
            continue;
        }


        std::vector<Point2i> pointCloud;
        for (auto &detectedLine : { line , *it }) {
            pointCloud.emplace_back(detectedLine[0], detectedLine[1]);
            pointCloud.emplace_back(detectedLine[2], detectedLine[3]);
        }

        line = HandlePointCloud(pointCloud);

        reducedLines.erase(it);
    }
}

} // namespace


std::vector<std::tuple<double, double, double, double, double>> calculating(const std::string& filename)
{
    Mat src = imread(filename, IMREAD_GRAYSCALE);

    if (src.empty()) {
        throw std::runtime_error("Error opening image");
    }

    Mat background;
    GaussianBlur(src, background, Size(63, 63), 0, 0, BORDER_DEFAULT);
    background -= 1;

    const auto kernel_size = 3;
    Mat dst;
    GaussianBlur(src, dst, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);
    const auto filtered = dst.clone();


    Mat stripeless;
    GaussianBlur(dst, stripeless, Size(63, 1), 0, 0, BORDER_DEFAULT);

    Mat func = dst - stripeless + 128;

    Mat funkyFunc;
    adaptiveThreshold(func, funkyFunc, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1.);

    Mat diff = filtered < background;


    // mask
    Mat mask;
    threshold(dst, mask, 180, 255, THRESH_BINARY_INV);
    int horizontal_size = 40;
    // Create structure element for extracting vertical lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
    // Apply morphology operations
    dilate(mask, mask, horizontalStructure);

    int circular_size = 5;
    Mat circularStructure = getStructuringElement(MORPH_ELLIPSE, Size(circular_size, circular_size));
    dilate(mask, mask, circularStructure);


    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 2);


    dst &= mask;
    dst &= diff;
    dst &= funkyFunc;


    cv::ximgproc::thinning(dst, dst);

    // Specify size on vertical axis
    int vertical_size = 4;
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    // Apply morphology operations
    erode(dst, dst, verticalStructure);
    dilate(dst, dst, verticalStructure);

    // Probabilistic Line Transform
    std::vector<Vec4i> linesP; // will hold the results of the detection
    const int threshold = 10;
    HoughLinesP(dst, linesP, 1, CV_PI / 180 / 10, threshold, 5, 25); // runs the actual detection

    linesP.erase(std::remove_if(linesP.begin(), linesP.end(), [&dst](const Vec4i& l) {
        const double expectedAlgle = 0.05;
        const auto border = 10;
        return l[1] == l[3] || fabs(double(l[0] - l[2]) / (l[1] - l[3]) + expectedAlgle) > expectedAlgle
            || l[0] < border && l[2] < border || l[1] == 0 && l[3] == 0
            || l[0] >= (dst.cols - border) && l[2] >= (dst.cols - border) || l[1] == dst.rows - 1 && l[3] == dst.rows - 1;
    }), linesP.end());


    auto reducedLines0 = reduceLines(linesP, 25, 1.0, 3);

    reducedLines0.erase(std::remove_if(reducedLines0.begin(), reducedLines0.end(), [](const Vec4i& line) {
        return hypot(line[2] - line[0], line[3] - line[1]) <= 10;
    }), reducedLines0.end());

    auto reducedLines = reduceLines(reducedLines0, 50, 0.7, 2.5);

    // find prevailing direction
    std::vector<Point2i> pointCloud;
    for (auto& reduced : reducedLines)
    {
        auto centerX = (reduced[0] + reduced[2]) / 2;
        auto centerY = (reduced[1] + reduced[3]) / 2;
        pointCloud.emplace_back(reduced[0] - centerX, reduced[1] - centerY);
        pointCloud.emplace_back(reduced[2] - centerX, reduced[3] - centerY);
    }
    Vec4f lineParams;
    fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);
    const auto cos_phi = -lineParams[1];
    const auto sin_phi = -lineParams[0];

    auto sortLam = [cos_phi, sin_phi](const Vec4i& detectedLine) {
        double x = (detectedLine[0] + detectedLine[2]) / 2.;
        double y = (detectedLine[1] + detectedLine[3]) / 2.;
        double x_new = x * cos_phi - y * sin_phi;
        return x_new;
    };

    std::sort(reducedLines.begin(), reducedLines.end(), [&sortLam](const Vec4i& l1, const Vec4i& l2) {
        return sortLam(l1) < sortLam(l2);
    });

    auto approveLam = [](const Vec4i& line) {
        return hypot(line[2] - line[0], line[3] - line[1]) > 70;
    };

    reducedLines.erase(reducedLines.begin(), std::find_if(reducedLines.begin(), reducedLines.end(), approveLam));
    reducedLines.erase(std::find_if(reducedLines.rbegin(), reducedLines.rend(), approveLam).base(), reducedLines.end());

    // merge
    MergeLines(reducedLines, sortLam);

    // normalize direction
    for (auto& line : reducedLines) {
        if (line[1] > line[3]) {
            std::swap(line[0], line[2]);
            std::swap(line[1], line[3]);
        }
    }


    std::vector<std::tuple<double, double, double, double, double>> result;

    for (int i = 0; i < reducedLines.size() - 1; ++i) {
        auto& first = reducedLines[i];
        auto& second = reducedLines[i + 1];
        if (first[1] > second[1]) {
            continue;
        }

        const auto y_first = first[0] * sin_phi + first[1] * cos_phi;
        const auto y_second = second[0] * sin_phi + second[1] * cos_phi;

        //const auto x_first = first[0] * cos_phi - first[1] * sin_phi;
        //const auto x_second = second[0] * cos_phi - second[1] * sin_phi;

        const auto diff = y_second - y_first;

        result.push_back({ first[0], first[1], second[0], second[1], diff });
    }

    return result;
}
