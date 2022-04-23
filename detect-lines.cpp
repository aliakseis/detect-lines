#include "detect-lines.h"

#include "known-good.h"

#include "tswdft2d.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/photo.hpp>

//#define USE_SURF

#ifdef USE_SURF
    #include <opencv2/xfeatures2d.hpp>
#endif

#include <ceres/ceres.h>

#include "nanoflann.hpp"

#include <array>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>

using namespace cv;

namespace {

using namespace cv;

// thinning stuff

enum ThinningTypes {
    THINNING_ZHANGSUEN = 0,  // Thinning technique of Zhang-Suen
    THINNING_GUOHALL = 1     // Thinning technique of Guo-Hall
};

// Applies a thinning iteration to a binary image
void thinningIteration(Mat img, int iter, int thinningType) {
    Mat marker = Mat::zeros(img.size(), CV_8UC1);

    if (thinningType == THINNING_ZHANGSUEN) {
        for (int i = 1; i < img.rows - 1; i++) {
            for (int j = 1; j < img.cols - 1; j++) {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int A = static_cast<int>(p2 == 0 && p3 == 1) + static_cast<int>(p3 == 0 && p4 == 1) +
                        static_cast<int>(p4 == 0 && p5 == 1) + static_cast<int>(p5 == 0 && p6 == 1) +
                        static_cast<int>(p6 == 0 && p7 == 1) + static_cast<int>(p7 == 0 && p8 == 1) +
                        static_cast<int>(p8 == 0 && p9 == 1) + static_cast<int>(p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
                    marker.at<uchar>(i, j) = 1;
                }
            }
        }
    }
    if (thinningType == THINNING_GUOHALL) {
        for (int i = 1; i < img.rows - 1; i++) {
            for (int j = 1; j < img.cols - 1; j++) {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int C = (static_cast<int>(p2 == 0u) & (p3 | p4)) + (static_cast<int>(p4 == 0u) & (p5 | p6)) +
                        (static_cast<int>(p6 == 0u) & (p7 | p8)) + (static_cast<int>(p8 == 0u) & (p9 | p2));
                int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
                int N = N1 < N2 ? N1 : N2;
                int m = iter == 0 ? ((p6 | p7 | static_cast<int>(p9 == 0u)) & p8) : ((p2 | p3 | static_cast<int>(p5 == 0u)) & p4);

                if ((C == 1) && ((N >= 2) && ((static_cast<int>((N <= 3)) & static_cast<int>(m == 0)) != 0))) {
                    marker.at<uchar>(i, j) = 1;
                }
            }
        }
    }

    img &= ~marker;
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output, int thinningType = THINNING_ZHANGSUEN) {
    Mat processed = input.getMat().clone();
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;

    Mat prev = Mat::zeros(processed.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(processed, 0, thinningType);
        thinningIteration(processed, 1, thinningType);
        absdiff(processed, prev, diff);
        processed.copyTo(prev);
    } while (countNonZero(diff) > 0);

    processed *= 255;

    output.assign(processed);
}

//////////////////////////////////////////////////////////////////////////////

// https://hbfs.wordpress.com/2018/03/13/paeths-method-square-roots-part-vii/
template <typename T>
auto fastHypot(T v1, T v2) {
    auto x = std::abs(v1);
    auto y = std::abs(v2);
    if (x < y) {
        std::swap(x, y);
    }
    if (x == 0) {
        return 0.;
    }
    auto y_x = static_cast<double>(y) / x;
    auto sq_y_x = y_x * y_x;
    return x * (1 + sq_y_x / 2 - (sq_y_x * sq_y_x) / 8);
}

void doFindPath(const cv::Mat& mat, const cv::Point& pt, cv::Point& final, int vertical, float cumulativeAngle,
                std::set<std::pair<int, int>>& passed = std::array<std::set<std::pair<int, int>>, 1>()[0]) {
    if (pt.x < 0 || pt.x >= mat.cols || pt.y < 0 || pt.y >= mat.rows) {
        return;
    }

    if (!passed.emplace(pt.x, pt.y).second) {
        return;
    }

    if (abs(vertical) > 1) {
        return;
    }

    if (fabs(cumulativeAngle) > 1.8) {
        return;
    }

    if (mat.at<uchar>(pt) == 0) {
        return;
    }

    if (final.y > pt.y) {
        final = pt;
    }

    cumulativeAngle *= 0.8;

    doFindPath(mat, Point(pt.x, pt.y - 1), final, 0, cumulativeAngle, passed);
    doFindPath(mat, Point(pt.x + 1, pt.y - 1), final, 0, cumulativeAngle + 0.5, passed);
    doFindPath(mat, Point(pt.x - 1, pt.y - 1), final, 0, cumulativeAngle - 0.5, passed);
    if (vertical >= 0) {
        doFindPath(mat, Point(pt.x + 1, pt.y), final, vertical + 1, cumulativeAngle + 1, passed);
    }
    if (vertical <= 0) {
        doFindPath(mat, Point(pt.x - 1, pt.y), final, vertical - 1, cumulativeAngle - 1, passed);
    }
}

cv::Point FindPath(const cv::Mat& mat, const cv::Point& start) {
    cv::Point pos = start;

    while (pos.x >= 0 && (mat.at<uchar>(pos) == 0 || (doFindPath(mat, pos, pos, 0, 0), pos.y == start.y))) {
        --pos.x;
    }

    if (pos.x < 0) {
        return start;
    }

    return pos;
}

//////////////////////////////////////////////////////////////////////////////

// https://stackoverflow.com/questions/30746327/get-a-single-line-representation-for-multiple-close-by-lines-clustered-together

auto extendedLine(const Vec4i& line, double d, double max_coeff) {
    const auto length = fastHypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = std::min(d / length, max_coeff);
    double xd = (line[2] - line[0]) * coeff;
    double yd = (line[3] - line[1]) * coeff;
    return Vec4f(line[0] - xd, line[1] - yd, line[2] + xd, line[3] + yd);
}

std::array<Point2f, 4> boundingRectangleContour(const Vec4i& line, float d) {
    // finds coordinates of perpendicular lines with length d in both line points
    const auto length = fastHypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = d / length;

    // dx:  -dy
    // dy:  dx
    double yd = (line[2] - line[0]) * coeff;
    double xd = -(line[3] - line[1]) * coeff;

    return {Point2f(line[0] - xd, line[1] - yd), Point2f(line[0] + xd, line[1] + yd), Point2f(line[2] + xd, line[3] + yd),
            Point2f(line[2] - xd, line[3] - yd)};
}

double pointPolygonTest_(const std::array<Point2f, 4>& contour, Point2f pt, bool measureDist) {

    double result = 0;
    int i;
    int total = contour.size();
    int counter = 0;

    double min_dist_num = FLT_MAX;
    double min_dist_denom = 1;

    const auto& cntf = contour;

    Point2f v0;
    Point2f v;

    v = cntf[total - 1];

    if (!measureDist) {
        for (i = 0; i < total; i++) {
            double dist;
            v0 = v;
            v = cntf[i];

            // if ((v0.y <= pt.y && v.y <= pt.y) || (v0.y > pt.y && v.y > pt.y) || (v0.x < pt.x && v.x < pt.x)) {
            if ((v0.y <= pt.y) == (v.y <= pt.y) || (v0.x < pt.x && v.x < pt.x)) {
                if (pt.y == v.y &&
                    (pt.x == v.x || (pt.y == v0.y && ((v0.x <= pt.x && pt.x <= v.x) || (v.x <= pt.x && pt.x <= v0.x))))) {
                    return 0;
                }
                continue;
            }

            dist = static_cast<double>(pt.y - v0.y) * (v.x - v0.x) - static_cast<double>(pt.x - v0.x) * (v.y - v0.y);
            if (dist == 0) {
                return 0;
            }
            if (v.y < v0.y) {
                dist = -dist;
            }
            counter += static_cast<int>(dist > 0);
        }

        result = counter % 2 == 0 ? -1 : 1;
    } else {
        for (i = 0; i < total; i++) {
            double dx;
            double dy;
            double dx1;
            double dy1;
            double dx2;
            double dy2;
            double dist_num;
            double dist_denom = 1;

            v0 = v;
            v = cntf[i];

            dx = v.x - v0.x;
            dy = v.y - v0.y;
            dx1 = pt.x - v0.x;
            dy1 = pt.y - v0.y;
            dx2 = pt.x - v.x;
            dy2 = pt.y - v.y;

            if (dx1 * dx + dy1 * dy <= 0) {
                dist_num = dx1 * dx1 + dy1 * dy1;
            } else if (dx2 * dx + dy2 * dy >= 0) {
                dist_num = dx2 * dx2 + dy2 * dy2;
            } else {
                dist_num = (dy1 * dx - dx1 * dy);
                dist_num *= dist_num;
                dist_denom = dx * dx + dy * dy;
            }

            if (dist_num * min_dist_denom < min_dist_num * dist_denom) {
                min_dist_num = dist_num;
                min_dist_denom = dist_denom;
                if (min_dist_num == 0) {
                    break;
                }
            }

            if ((v0.y <= pt.y && v.y <= pt.y) || (v0.y > pt.y && v.y > pt.y) || (v0.x < pt.x && v.x < pt.x)) {
                continue;
            }

            dist_num = dy1 * dx - dx1 * dy;
            if (dy < 0) {
                dist_num = -dist_num;
            }
            counter += static_cast<int>(dist_num > 0);
        }

        result = std::sqrt(min_dist_num / min_dist_denom);
        if (counter % 2 == 0) {
            result = -result;
        }
    }

    return result;
}

bool extendedBoundingRectangleLineEquivalence(const Vec4i& l1, const Vec4i& l2, float extensionLength,
                                              float extensionLengthMaxFraction, float boundingRectangleThickness) {

    const auto el1 = extendedLine(l1, extensionLength, extensionLengthMaxFraction);
    const auto el2 = extendedLine(l2, extensionLength, extensionLengthMaxFraction);

    // calculate window around extended line
    // at least one point needs to inside extended bounding rectangle of other line,
    const auto lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
    return pointPolygonTest_(lineBoundingContour, {el2[0], el2[1]}, false) >= 0 ||
           pointPolygonTest_(lineBoundingContour, {el2[2], el2[3]}, false) >= 0 ||

           pointPolygonTest_(lineBoundingContour, Point2f(l2[0], l2[1]), false) >= 0 ||
           pointPolygonTest_(lineBoundingContour, Point2f(l2[2], l2[3]), false) >= 0;
}

Vec4i HandlePointCloud(const std::vector<Point2i>& pointCloud) {
    // lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
    // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
    Vec4f lineParams;
    fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);

    // derive the bounding xs of point cloud
    std::vector<Point2i>::const_iterator minYP;
    std::vector<Point2i>::const_iterator maxYP;
    std::tie(minYP, maxYP) = std::minmax_element(pointCloud.begin(), pointCloud.end(),
                                                 [](const Point2i& p1, const Point2i& p2) { return p1.y < p2.y; });

    // derive y coords of fitted line
    float m = lineParams[0] / lineParams[1];
    int x1 = ((minYP->y - lineParams[3]) * m) + lineParams[2];
    int x2 = ((maxYP->y - lineParams[3]) * m) + lineParams[2];

    return {x1, minYP->y, x2, maxYP->y};
}

std::vector<Vec4i> reduceLines(const std::vector<Vec4i>& linesP, float extensionLength, float extensionLengthMaxFraction,
                               float boundingRectangleThickness) {
    // partition via our partitioning function
    std::vector<int> labels;
    int equilavenceClassesCount = cv::partition(
        linesP, labels,
        [extensionLength, extensionLengthMaxFraction, boundingRectangleThickness](const Vec4i& l1, const Vec4i& l2) {
            return extendedBoundingRectangleLineEquivalence(l1, l2,
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
        for (auto& detectedLine : groups[i]) {
            pointClouds[i].emplace_back(detectedLine[0], detectedLine[1]);
            pointClouds[i].emplace_back(detectedLine[2], detectedLine[3]);
        }
    }
    std::vector<Vec4i> reducedLines = std::accumulate(pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{},
                                                      [](std::vector<Vec4i> target, const std::vector<Point2i>& pointCloud) {
                                                          target.push_back(HandlePointCloud(pointCloud));
                                                          return target;
                                                      });

    return reducedLines;
}

template <typename T>
void MergeLines(std::vector<Vec4i>& reducedLines, T sortLam) {
    for (int i = reducedLines.size(); --i >= 0;) {
        auto& line = reducedLines[i];
        if (fastHypot(line[2] - line[0], line[3] - line[1]) > 30) {
            continue;
        }

        auto val = sortLam(line);

        double dist;
        std::vector<Vec4i>::iterator it;
        if (i == 0) {
            it = reducedLines.begin() + 1;
            dist = sortLam(*it) - val;
        } else if (i == reducedLines.size() - 1) {
            it = reducedLines.begin() + i - 2;
            dist = val - sortLam(*it);
        } else {
            const auto dist1 = val - sortLam(reducedLines[i - 1]);
            const auto dist2 = sortLam(reducedLines[i + 1]) - val;
            if (dist1 < dist2) {
                it = reducedLines.begin() + i - 1;
                dist = dist1;
            } else {
                it = reducedLines.begin() + i + 1;
                dist = dist2;
            }
        }

        const auto distY =
            abs((line[1] + line[3]) / 2 - ((*it)[1] + (*it)[3]) / 2) - (abs(line[1] - line[3]) + abs((*it)[1] - (*it)[3])) / 2;

        const auto threshold = 2.5;
        const auto thresholdY = 25;
        if (dist > threshold || distY > thresholdY) {
            reducedLines.erase(reducedLines.begin() + i);
            continue;
        }

        std::vector<Point2i> pointCloud;
        for (auto& detectedLine : {line, *it}) {
            pointCloud.emplace_back(detectedLine[0], detectedLine[1]);
            pointCloud.emplace_back(detectedLine[2], detectedLine[3]);
        }

        line = HandlePointCloud(pointCloud);

        reducedLines.erase(it);
    }
}

//////////////////////////////////////////////////////////////////////////////

const double POLY_COEFF = 0.001;

//////////////////////////////////////////////////////////////////////////////

double CalcPoly(const cv::Mat& X, double x) {
    double result = X.at<double>(0, 0);
    double v = 1.;
    for (int i = 1; i < X.rows; ++i) {
        v *= x;
        result += X.at<double>(i, 0) * v;
    }
    return result;
}

void fitLineRANSAC2(const std::vector<cv::Point>& vals, cv::Mat& a, int n_samples, std::vector<bool>& inlierFlag,
                    double noise_sigma = 5.) {
    int N = 5000;                // iterations
    double T = 3 * noise_sigma;  // residual threshold

    double max_weight = 0.;

    cv::Mat best_model(n_samples, 1, CV_64FC1);

    std::default_random_engine dre;

    std::vector<int> k(n_samples);

    for (int n = 0; n < N; n++) {
        // random sampling - n_samples points
        for (int j = 0; j < n_samples; ++j) {
            k[j] = j;
        }

        std::map<int, int> displaced;

        // Fisher-Yates shuffle Algorithm
        for (int j = 0; j < n_samples; ++j) {
            std::uniform_int_distribution<int> di(j, vals.size() - 1);
            int idx = di(dre);

            if (idx != j) {
                int& to_exchange = (idx < n_samples) ? k[idx] : displaced.try_emplace(idx, idx).first->second;
                std::swap(k[j], to_exchange);
            }
        }

        // model estimation
        cv::Mat AA(n_samples, n_samples, CV_64FC1);
        cv::Mat BB(n_samples, 1, CV_64FC1);
        for (int i = 0; i < n_samples; i++) {
            AA.at<double>(i, 0) = 1.;
            double v = 1.;
            for (int j = 1; j < n_samples; ++j) {
                v *= vals[k[i]].x * POLY_COEFF;
                AA.at<double>(i, j) = v;
            }

            BB.at<double>(i, 0) = vals[k[i]].y;
        }

        cv::Mat AA_pinv(n_samples, n_samples, CV_64FC1);
        invert(AA, AA_pinv, cv::DECOMP_SVD);

        cv::Mat X = AA_pinv * BB;

        // evaluation
        std::unordered_map<int, double> bestValues;
        double weight = 0.;
        for (const auto& v : vals) {
            const double arg = std::abs(v.y - CalcPoly(X, v.x * POLY_COEFF));
            const double data = exp(-arg * arg / (2 * noise_sigma * noise_sigma));

            auto& val = bestValues[v.x];
            if (data > val) {
                weight += data - val;
                val = data;
            }
        }

        if (weight > max_weight) {
            best_model = X;
            max_weight = weight;
        }
    }

    //------------------------------------------------------------------- optional LS fitting
    inlierFlag = std::vector<bool>(vals.size(), false);
    std::vector<int> vec_index;
    for (int i = 0; i < vals.size(); i++) {
        const auto& v = vals[i];
        double data = std::abs(v.y - CalcPoly(best_model, v.x * POLY_COEFF));
        if (data < T) {
            inlierFlag[i] = true;
            vec_index.push_back(i);
        }
    }

    cv::Mat A2(vec_index.size(), n_samples, CV_64FC1);
    cv::Mat B2(vec_index.size(), 1, CV_64FC1);

    for (int i = 0; i < vec_index.size(); i++) {
        A2.at<double>(i, 0) = 1.;
        double v = 1.;
        for (int j = 1; j < n_samples; ++j) {
            v *= vals[vec_index[i]].x * POLY_COEFF;
            A2.at<double>(i, j) = v;
        }

        B2.at<double>(i, 0) = vals[vec_index[i]].y;
    }

    cv::Mat A2_pinv(n_samples, vec_index.size(), CV_64FC1);
    invert(A2, A2_pinv, cv::DECOMP_SVD);

    a = A2_pinv * B2;
}

//////////////////////////////////////////////////////////////////////////////

struct PolynomialResidual {
    PolynomialResidual(double x, double y, int n_samples) : x_(x), y_(y), n_samples_(n_samples) {}

    template <typename T>
    bool operator()(T const* const* relative_poses, T* residuals) const {

        T y = *(relative_poses[0]) + *(relative_poses[1]) * x_;
        for (int i = 2; i < n_samples_; ++i) {
            y += *(relative_poses[i]) * std::pow(x_, i);
        }

        residuals[0] = T(y_) - y;
        return true;
    }

   private:
    // Observations for a sample.
    const double x_;
    const double y_;
    int n_samples_;
};

//////////////////////////////////////////////////////////////////////////////

class PointsProvider {
   public:
    PointsProvider(const std::vector<cv::Point>& ptSet) : ptSet_(ptSet) {}

    size_t kdtree_get_point_count() const { return ptSet_.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    float kdtree_get_pt(const size_t idx, const size_t dim) const {
        auto& v = ptSet_[idx];
        return dim != 0u ? v.y : v.x;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }

   private:
    const std::vector<cv::Point>& ptSet_;
};  // namespace

// construct a kd-tree index:
using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointsProvider>, PointsProvider, 2>;

float FindThresholdIIntensity(const cv::Mat& dst) {
    std::priority_queue<float, std::vector<float>, std::greater<>> heap;
    const auto HEAP_SIZE = dst.rows * dst.cols * 2 / 5;
    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            auto v = dst.at<float>(y, x);
            if (heap.size() >= HEAP_SIZE) {
                if (heap.top() >= v) {
                    continue;
                }
                heap.pop();
            }
            heap.push(v);
        }
    }

    return heap.top();
}

//////////////////////////////////////////////////////////////////////////////

const int IMAGE_DIMENSION = 800;

enum { WINDOW_DIMENSION_X = 64 };
enum { WINDOW_DIMENSION_Y = 1 };

const auto visualizationRows = IMAGE_DIMENSION - WINDOW_DIMENSION_Y + 1;
const auto visualizationCols = IMAGE_DIMENSION - WINDOW_DIMENSION_X + 1;

//////////////////////////////////////////////////////////////////////////////

auto groupByFourier(cv::Mat img, cv::Mat mask) {

    GaussianBlur(img, img, cv::Size(1, 33), 0, 0);

    std::vector<unsigned char> freqs(visualizationRows * visualizationCols);

    cv::Mat amplitudes(visualizationRows, visualizationCols, CV_32FC1);

    {
        double amplitudeCoeffs[WINDOW_DIMENSION_X];
        amplitudeCoeffs[0] = 0;
        for (int i = 1; i < WINDOW_DIMENSION_X; ++i) {
            amplitudeCoeffs[i] = 1. / sqrt(sqrt(i));
        }

        auto lam = [&](const Range& range) {
            auto transformed = tswdft2d((reinterpret_cast<float*>(img.data)) + img.cols * range.start, WINDOW_DIMENSION_Y,
                                        WINDOW_DIMENSION_X, range.end - range.start, img.cols);

            for (int y = range.start; y < range.end; ++y) {
                for (int x = 0; x < visualizationCols; ++x) {
                    const auto sourceOffset = (y - range.start) * visualizationCols + x;
                    const auto destinationOffset = y * visualizationCols + x;

                    unsigned int freq = 0;

                    float threshold = 0;

                    for (unsigned int j = 3; j <= WINDOW_DIMENSION_X / 2; ++j) {
                        const auto& v = transformed[sourceOffset * WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X + j];
                        const auto amplitude = fastHypot(v.real(), v.imag()) * amplitudeCoeffs[j];
                        if (amplitude > threshold) {
                            freq = j;
                            threshold = amplitude;
                        }
                    }
                    freqs[destinationOffset] = freq;
                    amplitudes.at<float>(y, x) = threshold;
                }
            }
        };

        std::vector<std::future<void>> proxies;

        const auto numChunks = visualizationRows / 2;

        for (int i = 0; i < numChunks; ++i) {
            proxies.push_back(std::async(std::launch::async, lam,
                                         Range((visualizationRows * i) / numChunks, (visualizationRows * (i + 1)) / numChunks)));
        }
    }

    amplitudes += 1.;
    cv::log(amplitudes, amplitudes);
    cv::normalize(amplitudes, amplitudes, 0, 1, cv::NORM_MINMAX);

    cv::Mat borderline00(visualizationRows, visualizationCols, CV_8UC1, cv::Scalar(0));
    cv::Mat borderline0(visualizationRows, visualizationCols, CV_8UC1, cv::Scalar(0));

    // border line
    std::vector<cv::Point> ptSet;

    for (int gloriousAttempt = 0; gloriousAttempt < 2; ++gloriousAttempt) {
        std::vector<int> lastTransitions(visualizationCols, INT_MIN / 2);
        for (int yy = 0; yy < visualizationRows - 1; ++yy) {
            for (int x = 0; x < visualizationCols; ++x) {
                const int y = gloriousAttempt != 0 ? (visualizationRows - 1 - yy) : yy;

                const auto sourceOffset1 = y * visualizationCols + x;
                const auto sourceOffset2 =
                    (gloriousAttempt != 0 ? (visualizationRows - 2 - yy) : (yy + 1)) * visualizationCols + x;
                int freq1 = freqs[sourceOffset1];
                int freq2 = freqs[sourceOffset2];

                enum { PROBE_BIAS = 10 };
                // if (freq1 > 2 && freq1 >= ((freq2 * 3 / 5 - 1)) && freq1 <= ((freq2 * 3 / 5 + 1)))
                if (y > PROBE_BIAS && freq2 > freq1 && freq2 >= freq1 * 5 / 3 && freq2 <= freq1 * 3)  // 5 / 2)
                {
                    const auto maskOk =
                        (mask.at<uchar>(y + WINDOW_DIMENSION_Y / 2, x + WINDOW_DIMENSION_X / 2) != 0u) &&
                        (mask.at<uchar>(y + WINDOW_DIMENSION_Y / 2 - PROBE_BIAS, x + WINDOW_DIMENSION_X / 2) != 0u);

                    if (maskOk && yy - lastTransitions[x] > 50) {
                        lastTransitions[x] = yy;
                        if (y < visualizationRows - 100) {  // exclude lowest area
                            borderline00.at<uchar>(y, x) = 255;
                            ptSet.emplace_back(x, y);
                        }
                    }
                }
            }
        }
    }
    // filtering
    for (auto& pt : ptSet) {
        pt.y *= 2;  // introduce anisotropy
    }

    for (;;) {
        PointsProvider provider(ptSet);

        my_kd_tree_t infos(2, provider);

        infos.buildIndex();

        const int k = 16;

        std::vector<size_t> index(k);
        std::vector<float> dist(k);

        std::vector<bool> goodOnes(ptSet.size());

        for (int i = 0; i < ptSet.size(); ++i) {

            float pos[2];

            pos[0] = ptSet[i].x;
            pos[1] = ptSet[i].y;

            infos.knnSearch(&pos[0], k, &index[0], &dist[0]);

            goodOnes[i] = dist[k - 1] < 30 * 30;
        }

        bool found = false;
        for (int i = ptSet.size(); --i >= 0;) {
            if (!goodOnes[i]) {
                found = true;
                ptSet.erase(ptSet.begin() + i);
            }
        }
        if (!found) {
            break;
        }
    }

    for (auto& pt : ptSet) {
        pt.y /= 2;
    }

    // partition via our partitioning function
    std::vector<int> labels;
    int equilavenceClassesCount = cv::partition(
        ptSet, labels, [](const cv::Point& p1, const cv::Point& p2) { return fastHypot(p2.x - p1.x, p2.y - p1.y) < 25; });

    std::vector<int> groupCounts(equilavenceClassesCount);
    std::vector<int> groupYCounts(equilavenceClassesCount);

    for (int i = ptSet.size(); --i >= 0;) {
        auto l = labels[i];
        auto& pt = ptSet[i];

        ++groupCounts[l];
        groupYCounts[l] += pt.y;
    }

    // merge
    enum { MERGE_VARIANCE = 10 };
    for (int l2 = equilavenceClassesCount; --l2 > 0;) {
        for (int l1 = l2; --l1 >= 0;) {
            if (std::abs(double(groupYCounts[l1]) / groupCounts[l1] - double(groupYCounts[l2]) / groupCounts[l2]) <
                MERGE_VARIANCE) {
                groupCounts[l1] += groupCounts[l2];
                groupCounts.erase(groupCounts.begin() + l2);
                groupYCounts[l1] += groupYCounts[l2];
                groupYCounts.erase(groupYCounts.begin() + l2);

                --equilavenceClassesCount;

                for (auto& l : labels) {
                    if (l == l2) {
                        l = l1;
                    } else if (l > l2) {
                        --l;
                    }
                }
                break;
            }
        }
    }

    return std::make_tuple(equilavenceClassesCount, ptSet, labels);
}

//////////////////////////////////////////////////////////////////////////////

enum { n_samples = 8 };

auto polySolve(const std::vector<cv::Point>& ptSet) {

    // putenv("GLOG_logtostderr=1");
    // putenv("GLOG_stderrthreshold=3");
    // putenv("GLOG_minloglevel=3");
    // putenv("GLOG_v=-3");

    auto ransacLam = [](const std::vector<cv::Point>& ptSet, int n_ransac_samples) {
        cv::Mat A;
        std::vector<bool> inliers;
        fitLineRANSAC2(ptSet, A, n_ransac_samples,  // A, B, C,
                       inliers);

        for (int i = 0; i < n_samples - n_ransac_samples; ++i) {
            A.push_back(0.);
        }

        std::vector<double*> params;
        params.reserve(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            params.push_back(&A.at<double>(i, 0));
        }

        ceres::Problem problem;
        for (auto& i : ptSet) {
            auto cost_function = new ceres::DynamicAutoDiffCostFunction<PolynomialResidual>(
                new PolynomialResidual(i.x * POLY_COEFF, i.y, n_samples));

            for (int j = 0; j < params.size(); ++j) {
                cost_function->AddParameterBlock(1);
            }

            cost_function->SetNumResiduals(1);

            problem.AddResidualBlock(cost_function, new ceres::ArctanLoss(5.), params);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        options.max_num_iterations = 1000;

        options.logging_type = ceres::SILENT;

        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);

        return std::make_pair(summary.final_cost, A);
    };

    std::vector<std::future<decltype(ransacLam({}, 0))>> proxies;

    for (int n_ransac_samples = 1; n_ransac_samples <= n_samples; ++n_ransac_samples) {
        proxies.push_back(std::async(std::launch::async, ransacLam, ptSet, n_ransac_samples));
    }

    double bestCost = 1.e38;

    cv::Mat poly;

    for (auto& p : proxies) {
        auto v = p.get();
        if (v.first < bestCost) {
            bestCost = v.first;
            poly = v.second;
        }
    }

    return poly;
}

}  // namespace

void imaging::calcGST(const cv::Mat& inputImg, cv::Mat& imgCoherencyOut, cv::Mat& imgOrientationOut, int w /*= 52*/) {
    using namespace cv;

    Mat img;
    inputImg.convertTo(img, CV_32F);
    // GST components calculation (start)
    // J =  (J11 J12; J12 J22) - GST
    Mat imgDiffX;
    Mat imgDiffY;
    Mat imgDiffXY;
    Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
    Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
    multiply(imgDiffX, imgDiffY, imgDiffXY);
    Mat imgDiffXX;
    Mat imgDiffYY;
    multiply(imgDiffX, imgDiffX, imgDiffXX);
    multiply(imgDiffY, imgDiffY, imgDiffYY);
    Mat J11;
    Mat J22;
    Mat J12;  // J11, J22 and J12 are GST components
    boxFilter(imgDiffXX, J11, CV_32F, Size(w, w));
    boxFilter(imgDiffYY, J22, CV_32F, Size(w, w));
    boxFilter(imgDiffXY, J12, CV_32F, Size(w, w));
    // GST components calculation (stop)
    // eigenvalue calculation (start)
    // lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
    // lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
    Mat tmp1;
    Mat tmp2;
    Mat tmp3;
    Mat tmp4;
    tmp1 = J11 + J22;
    tmp2 = J11 - J22;
    multiply(tmp2, tmp2, tmp2);
    multiply(J12, J12, tmp3);
    sqrt(tmp2 + 4.0 * tmp3, tmp4);
    Mat lambda1;
    Mat lambda2;
    lambda1 = tmp1 + tmp4;
    lambda1 = 0.5 * lambda1;  // biggest eigenvalue
    lambda2 = tmp1 - tmp4;
    lambda2 = 0.5 * lambda2;  // smallest eigenvalue
    // eigenvalue calculation (stop)
    // Coherency calculation (start)
    // Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    // Coherency is anisotropy degree (consistency of local orientation)
    divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherencyOut);
    // Coherency calculation (stop)
    // orientation angle calculation (start)
    // tan(2*Alpha) = 2*J12/(J22 - J11)
    // Alpha = 0.5 atan2(2*J12/(J22 - J11))
    phase(J22 - J11, 2.0 * J12, imgOrientationOut, false);
    imgOrientationOut = 0.5 * imgOrientationOut;
    // orientation angle calculation (stop)
}

//////////////////////////////////////////////////////////////////////////////

std::vector<std::tuple<double, double, double, double, double>> imaging::calculating(
    const std::string& filename, std::function<void(const cv::String&, cv::InputArray)> do_imshow_) {

    const bool do_imshow = do_imshow_ != nullptr;

    auto start = std::chrono::high_resolution_clock::now();

    Mat src = imread(filename, cv::IMREAD_ANYDEPTH | IMREAD_GRAYSCALE);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);

    std::cout << duration.count() << " microseconds.\n";

    if (src.empty()) {
        throw std::runtime_error("Error opening image");
    }

    const Size originalSize(src.cols, src.rows);

    cv::Mat img;
    src.convertTo(img, CV_32F);

    if ((src.type() & CV_MAT_DEPTH_MASK) == CV_16U) {
        img /= 256.;
    }

    cv::resize(img, img, cv::Size(IMAGE_DIMENSION, IMAGE_DIMENSION), 0, 0, cv::INTER_LANCZOS4);

    Scalar m = cv::mean(img);
    std::cout << "*** Mean value: " << m << '\n';

    auto ms = moments(img);
    const double base = ms.m00 * (IMAGE_DIMENSION - 1.) / 2;
    const bool mirrorX = ms.m10 > base;
    const bool mirrorY = ms.m01 > base;

    if (mirrorX) {
        flip(img, img, 1);
    }
    if (mirrorY) {
        flip(img, img, 0);
    }

    auto imshow = [do_imshow_, mirrorX, mirrorY](const char* winname, cv::Mat& img) {
        if (do_imshow_) {
            if (mirrorX) {
                flip(img, img, 1);
            }
            if (mirrorY) {
                flip(img, img, 0);
            }

            if (img.cols < IMAGE_DIMENSION || img.rows < IMAGE_DIMENSION) {
                cv::Mat dst;
                copyMakeBorder(img, dst, (IMAGE_DIMENSION - img.rows) / 2, (IMAGE_DIMENSION - img.rows) / 2,
                               (IMAGE_DIMENSION - img.cols) / 2, (IMAGE_DIMENSION - img.cols) / 2, BORDER_REPLICATE);
                do_imshow_(winname, dst);
            } else {
                do_imshow_(winname, img);
            }

            if (mirrorX) {
                flip(img, img, 1);
            }
            if (mirrorY) {
                flip(img, img, 0);
            }
        }
    };

    const auto kernel_size = 3;
    cv::Mat dst;
    GaussianBlur(img, dst, cv::Size(kernel_size, kernel_size), 0, 0);
    const auto filtered = dst.clone();

    Mat background;
    GaussianBlur(img, background, Size(63, 11), 0, 0);

    Mat diff = dst < background;

    imshow("Diff", diff);

    // median stuff
    const float thr = FindThresholdIIntensity(dst);
    Mat mask = dst < thr;

    imshow("Mask", mask);

    dst += 1.;
    cv::log(dst, dst);

    auto tomasiLam = [](cv::Mat dst) {
        cv::Mat tomasiRspImg = Mat::zeros(dst.size(), CV_32FC1);
        // Calculate the minimum eigenvalue, namely ShiTomasi angle response value R = min(L1, L2)
        int blockSize = 3;
        int ksize = 3;
        cv::cornerMinEigenVal(dst, tomasiRspImg, blockSize, ksize, BORDER_DEFAULT);
        return tomasiRspImg;
    };
    auto tomasiFut = std::async(std::launch::async, tomasiLam, dst.clone()).share();

    cv::Mat stripeless;
    GaussianBlur(dst, stripeless, cv::Size(63, 1), 0, 0);

    cv::Mat funcFloat = dst - stripeless;
    normalize(funcFloat, funcFloat, -64, 255 + 64, cv::NORM_MINMAX);
    cv::Mat func;
    funcFloat.convertTo(func, CV_8U);

    // !!!
    dst = func.clone();

    GaussianBlur(dst, dst, cv::Size(1, 5), 0, 0);

    imshow("dst filtered", dst);

#ifdef USE_SURF
    auto surfLam = [&func] {
        auto surf = cv::xfeatures2d::SURF::create(1700);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        surf->detectAndCompute(func, cv::noArray(), keypoints, descriptors);

        // http://itnotesblog.ru/note.php?id=271
        cv::FlannBasedMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors, GetKnownGood(), matches);
        return std::make_pair(keypoints, matches);
    };

    auto surfFut = std::async(std::launch::async, surfLam).share();
#endif

    //////////////////////////////////////////////////////////////////////////////

    cv::Mat imgCoherency;
    cv::Mat imgOrientation;
    calcGST(funcFloat, imgCoherency, imgOrientation);

    cv::Mat imgOrientationBin;
    inRange(imgOrientation, cv::Scalar(CV_PI / 2 - 0.2), cv::Scalar(CV_PI / 2 + 0.2), imgOrientationBin);

    cv::Mat imgCoherencyBin = imgCoherency > 0.28;

    auto [equilavenceClassesCount, ptSet, labels] = groupByFourier(img, mask & imgCoherencyBin & imgOrientationBin);
    {
        std::vector<int> groupCounts(equilavenceClassesCount);
        std::vector<int> groupXCounts(equilavenceClassesCount);
        std::vector<int> groupSqXCounts(equilavenceClassesCount);
        std::vector<int> groupYCounts(equilavenceClassesCount);
        std::vector<int> groupSqYCounts(equilavenceClassesCount);

        std::vector<float> groupAccumCoherencies(equilavenceClassesCount);

        for (int i = ptSet.size(); --i >= 0;) {
            auto l = labels[i];
            auto& pt = ptSet[i];

            ++groupCounts[l];
            groupXCounts[l] += pt.x;
            groupSqXCounts[l] += pt.x * pt.x;
            groupYCounts[l] += pt.y;
            groupSqYCounts[l] += pt.y * pt.y;

            groupAccumCoherencies[l] += imgCoherency.at<float>(pt.y, pt.x);
        }

        auto maxIdx = std::max_element(groupCounts.begin(), groupCounts.end()) - groupCounts.begin();
        const auto threshold = groupCounts[maxIdx] * 0.4;

        for (int i = 0; i < equilavenceClassesCount; ++i) {
            groupAccumCoherencies[i] /= groupCounts[i];
        }

        const auto coherencyThreshold = *std::max_element(groupAccumCoherencies.begin(), groupAccumCoherencies.end()) * 0.7;

        auto lam = [&](int l) {
            auto avgX = double(groupXCounts[l]) / groupCounts[l];
            auto avgY = double(groupYCounts[l]) / groupCounts[l];
            auto devX = sqrt(double(groupSqXCounts[l]) / groupCounts[l] - avgX * avgX);
            auto devY = sqrt(double(groupSqYCounts[l]) / groupCounts[l] - avgY * avgY);
            return devY / (devX + devY);
        };

        auto bestSlope = lam(maxIdx);
        auto slopeThreshold = std::min(bestSlope * 4.2, std::max(bestSlope, 0.3));

        double maxY = 0;

        for (int i = ptSet.size(); --i >= 0;) {
            auto l = labels[i];
            if (groupCounts[l] < threshold || lam(l) > slopeThreshold || groupAccumCoherencies[l] < coherencyThreshold) {
                ptSet.erase(ptSet.begin() + i);
                labels.erase(labels.begin() + i);
            } else {
                double y = double(groupYCounts[l]) / groupCounts[l];
                if (y > maxY) {
                    maxY = y;
                }
            }
        }

        for (int i = ptSet.size(); --i >= 0;) {
            auto l = labels[i];
            double y = double(groupYCounts[l]) / groupCounts[l];
            if (maxY - y > 50) {
                ptSet.erase(ptSet.begin() + i);
                labels.erase(labels.begin() + i);
            }
        }
    }

    if (ptSet.empty()) {
        imshow("image", func);
        return {};
    }

    cv::Mat poly = polySolve(ptSet);

    int x_min = INT_MAX;
    int x_max = INT_MIN;
    // limits
    for (auto& v : ptSet) {
        auto y = CalcPoly(poly, v.x * POLY_COEFF);
        if (fabs(y - v.y) < 15) {
            x_min = std::min(x_min, v.x);
            x_max = std::max(x_max, v.x);
        }
    }

    std::vector<cv::KeyPoint> goodkeypoints;

    double tomasi_min_rsp;
    double tomasi_max_rsp;
    auto tomasiRspImg = tomasiFut.get();

    if (do_imshow) {
        auto tomasiRspDisplay = tomasiRspImg.clone();
        cv::normalize(tomasiRspDisplay, tomasiRspDisplay, 0, 255, cv::NORM_MINMAX);
        cv::Mat dst;
        tomasiRspDisplay.convertTo(dst, CV_8U);
        adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 3.);
        imshow("tomasiRspDisplay", dst);
    }

    // Find the maximum and minimum angle response of tomasiRspImg, and their location, pass 0 for position parameter
    minMaxLoc(tomasiRspImg, &tomasi_min_rsp, &tomasi_max_rsp, nullptr, nullptr, Mat());
    const double tomasi_coeff = 0.02;
    const float tomasi_t = tomasi_min_rsp + tomasi_coeff * (tomasi_max_rsp - tomasi_min_rsp);

#ifdef USE_SURF
    auto [keypoints, matches] = surfFut.get();

    for (int i = 0; i < keypoints.size(); i++) {
        if (keypoints[i].size < 5 || keypoints[i].size > 50) continue;

        enum { HALF_SIZE = 5 };
        float v = tomasi_min_rsp;
        for (int y = std::max(int(keypoints[i].pt.y + 0.5) - HALF_SIZE, 0);
             y <= std::min(int(keypoints[i].pt.y + 0.5) + HALF_SIZE, tomasiRspImg.rows - 1); ++y)
            for (int x = std::max(int(keypoints[i].pt.x + 0.5) - HALF_SIZE, 0);
                 x <= std::min(int(keypoints[i].pt.x + 0.5) + HALF_SIZE, tomasiRspImg.cols - 1); ++x)
                v = std::max(v, tomasiRspImg.at<float>(y, x));

        if (v < tomasi_t) continue;

        if (matches[i].distance < 0.33) {
            double y =
                CalcPoly(poly, std::clamp(keypoints[i].pt.x - WINDOW_DIMENSION_X / 2, float(x_min), float(x_max)) * POLY_COEFF) +
                WINDOW_DIMENSION_Y / 2;
            if (fabs(y - keypoints[i].pt.y) < 50) goodkeypoints.push_back(keypoints[i]);
        }
    }
#endif

    for (int i = goodkeypoints.size() - 1; --i >= 0;) {
        for (int j = goodkeypoints.size(); --j > i;) {
            if (fastHypot(goodkeypoints[i].pt.x - goodkeypoints[j].pt.x, goodkeypoints[i].pt.y - goodkeypoints[j].pt.y) < 5) {
                goodkeypoints.erase(goodkeypoints.begin() + j);
            }
        }
    }

    std::vector<cv::Point> points_fitted;
    for (int x = 0; x < visualizationCols; x++) {
        double y = CalcPoly(poly, std::clamp(x, x_min, x_max) * POLY_COEFF);
        points_fitted.emplace_back(x + WINDOW_DIMENSION_X / 2, y + WINDOW_DIMENSION_Y / 2);
    }

    if (do_imshow) {
        cv::polylines(func, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
        imshow("image", func);
    }

    ///////////////////////////////////////////////////////////////////////////////

    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 3.);

    imshow("Dst before 000", dst);

    dst &= imgCoherencyBin;

    dst &= imgOrientationBin;

    dst &= mask;

    dst &= diff;

    imshow("Dst before", dst);

    auto thinningLam = [](cv::Mat src) {
        cv::Mat result;
        thinning(src, result);
        return result;
    };

    enum { THINNING_MARGIN = 50 };
    auto thinningFut00 = std::async(std::launch::async, thinningLam,
                                    dst(cv::Rect(0, 0, dst.cols / 2 + THINNING_MARGIN, dst.rows / 2 + THINNING_MARGIN)));

    auto thinningFut01 = std::async(
        std::launch::async, thinningLam,
        dst(cv::Rect(0, dst.rows / 2 - THINNING_MARGIN, dst.cols / 2 + THINNING_MARGIN, dst.rows / 2 + THINNING_MARGIN)));

    auto thinningFut10 = std::async(
        std::launch::async, thinningLam,
        dst(cv::Rect(dst.cols / 2 - THINNING_MARGIN, 0, dst.cols / 2 + THINNING_MARGIN, dst.rows / 2 + THINNING_MARGIN)));

    auto thinningFut11 = std::async(std::launch::async, thinningLam,
                                    dst(cv::Rect(dst.cols / 2 - THINNING_MARGIN, dst.rows / 2 - THINNING_MARGIN,
                                                 dst.cols / 2 + THINNING_MARGIN, dst.rows / 2 + THINNING_MARGIN)));

    // auto skeleton = dst.clone();
    cv::Mat skeleton(dst.rows, dst.cols, CV_8UC1);
    thinningFut00.get()(cv::Rect(0, 0, dst.cols / 2, dst.rows / 2)).copyTo(skeleton(cv::Rect(0, 0, dst.cols / 2, dst.rows / 2)));

    thinningFut01.get()(cv::Rect(0, THINNING_MARGIN, dst.cols / 2, dst.rows / 2))
        .copyTo(skeleton(cv::Rect(0, dst.rows / 2, dst.cols / 2, dst.rows / 2)));

    thinningFut10.get()(cv::Rect(THINNING_MARGIN, 0, dst.cols / 2, dst.rows / 2))
        .copyTo(skeleton(cv::Rect(dst.cols / 2, 0, dst.cols / 2, dst.rows / 2)));

    thinningFut11.get()(cv::Rect(THINNING_MARGIN, THINNING_MARGIN, dst.cols / 2, dst.rows / 2))
        .copyTo(skeleton(cv::Rect(dst.cols / 2, dst.rows / 2, dst.cols / 2, dst.rows / 2)));

    dst = skeleton.clone();

    imshow("Thinning", dst);

    // Specify size on vertical axis
    int vertical_size = 5;  // dst.rows / 30;
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    // Apply morphology operations
    erode(dst, dst, verticalStructure);
    dilate(dst, dst, verticalStructure);

    for (int y = 0; y < dst.rows - 4; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            if ((dst.at<uchar>(y, x) != 0u) && (dst.at<uchar>(y + 3, x) != 0u)) {
                dst.at<uchar>(y + 1, x) = 127;
                dst.at<uchar>(y + 2, x) = 127;
            }
        }
    }

    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            if (dst.at<uchar>(y, x) == 127) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }

    for (int y = 1; y < dst.rows - 1; ++y) {
        for (int x = 1; x < dst.cols - 1; ++x) {
            if (skeleton.at<uchar>(y, x) == 255 && dst.at<uchar>(y, x) == 0 &&
                (dst.at<uchar>(y - 1, x - 1) == 255 || dst.at<uchar>(y + 1, x - 1) == 255 || dst.at<uchar>(y - 1, x + 1) == 255 ||
                 dst.at<uchar>(y + 1, x + 1) == 255 || dst.at<uchar>(y, x - 1) == 255 || dst.at<uchar>(y, x + 1) == 255)) {
                dst.at<uchar>(y, x) = 127;
            }
        }
    }

    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            if (dst.at<uchar>(y, x) == 127) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }

    // x
    for (int y = 0; y < dst.rows - 1; ++y) {
        for (int x = 0; x < dst.cols - 1; ++x) {
            if (dst.at<uchar>(y, x) == 255 && dst.at<uchar>(y + 1, x + 1) == 255 && dst.at<uchar>(y, x + 1) == 0 &&
                dst.at<uchar>(y + 1, x) == 0) {
                dst.at<uchar>(y, x + 1) = 127;
                dst.at<uchar>(y + 1, x) = 127;
            } else if (dst.at<uchar>(y, x) == 0 && dst.at<uchar>(y + 1, x + 1) == 0 && dst.at<uchar>(y, x + 1) == 255 &&
                       dst.at<uchar>(y + 1, x) == 255) {
                dst.at<uchar>(y, x) = 127;
                dst.at<uchar>(y + 1, x + 1) = 127;
            }
        }
    }

    for (int y = 0; y < dst.rows; ++y) {
        for (int x = 0; x < dst.cols; ++x) {
            if (dst.at<uchar>(y, x) == 127) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }

    //*
    auto houghLinesLam = [](const cv::Mat& dst) {
        std::vector<cv::Vec4i> linesP;  // will hold the results of the detection
        const int threshold = 8;
        HoughLinesP(dst, linesP, 0.5, CV_PI / 180 / 2, threshold, 3, 25);  // runs the actual detection
        return linesP;
    };

    enum { HOUGH_THINNING_MARGIN = 50 };
    auto houghLinesFut0 =
        std::async(std::launch::async, houghLinesLam, dst(cv::Rect(0, 0, dst.cols / 2 + HOUGH_THINNING_MARGIN, dst.rows)));
    auto houghLinesFut1 =
        std::async(std::launch::async, houghLinesLam,
                   dst(cv::Rect(dst.cols / 2 - HOUGH_THINNING_MARGIN, 0, dst.cols / 2 + HOUGH_THINNING_MARGIN, dst.rows)));

    auto linesP = houghLinesFut0.get();

    linesP.erase(std::remove_if(linesP.begin(), linesP.end(),
                                [cols = dst.cols](const Vec4i& l) { return l[0] > cols / 2 && l[2] > cols / 2; }),
                 linesP.end());

    {
        auto linesP1 = houghLinesFut1.get();

        linesP1.erase(std::remove_if(linesP1.begin(), linesP1.end(),
                                     [](const Vec4i& l) { return l[0] < HOUGH_THINNING_MARGIN && l[2] < HOUGH_THINNING_MARGIN; }),
                      linesP1.end());

        for (auto& l : linesP1) {
            l[0] += dst.cols / 2 - HOUGH_THINNING_MARGIN;
            l[2] += dst.cols / 2 - HOUGH_THINNING_MARGIN;
        }

        linesP.insert(linesP.end(), linesP1.begin(), linesP1.end());
    }
    //*/

    linesP.erase(std::remove_if(linesP.begin(), linesP.end(),
                                [&dst](const Vec4i& l) {
                                    const double expectedAlgle = 0;
                                    const double expectedAngleDiff = 1.;
                                    const auto border = 10;  // gloriousAttempt ? 50 : 10;
                                    return l[1] == l[3] ||
                                           fabs(double(l[0] - l[2]) / (l[1] - l[3]) + expectedAlgle) > expectedAngleDiff ||
                                           l[0] < border && l[2] < border || l[1] == 0 && l[3] == 0 ||
                                           l[0] >= (dst.cols - border) && l[2] >= (dst.cols - border) ||
                                           l[1] == dst.rows - 1 && l[3] == dst.rows - 1;
                                }),
                 linesP.end());

    if (linesP.empty()) {
        return {};
    }

    {
        // partition via our partitioning function
        std::vector<int> labels;
        int equilavenceClassesCount = cv::partition(linesP, labels, [](const Vec4i& l1, const Vec4i& l2) {
            const auto distX =
                std::max(abs((l1[0] + l1[2]) / 2 - (l2[0] + l2[2]) / 2) - (abs(l1[0] - l1[2]) + abs(l2[0] - l2[2])) / 2, 0);
            const auto distY =
                std::max(abs((l1[1] + l1[3]) / 2 - (l2[1] + l2[3]) / 2) - (abs(l1[1] - l1[3]) + abs(l2[1] - l2[3])) / 2, 0);
            return fastHypot(distX / 25., distY / 15.) < 1;
        });

        std::vector<int> groupCounts(equilavenceClassesCount);
        for (auto& l : labels) {
            ++groupCounts[l];
        }

        const auto threshold = *std::max_element(groupCounts.begin(), groupCounts.end()) * 0.2;
        for (int i = linesP.size(); --i >= 0;) {
            if (groupCounts[labels[i]] < threshold) {
                linesP.erase(linesP.begin() + i);
            }
        }
    }

    auto angleSortLam = [](const Vec4i& l) { return double(l[0] - l[2]) / (l[1] - l[3]); };

    std::sort(linesP.begin(), linesP.end(),
              [&angleSortLam](const Vec4i& l1, const Vec4i& l2) { return angleSortLam(l1) < angleSortLam(l2); });

    const double maxDiff = 0.1;

    auto itFirst = linesP.begin();
    auto itLast = linesP.begin();

    double sum = 0;
    double maxSum = 0;

    auto itBegin = linesP.begin();
    auto itEnd = linesP.begin();

    while (itFirst != linesP.end() && itLast != linesP.end()) {
        auto start = angleSortLam(*itFirst);

        while (itLast != linesP.end() && angleSortLam(*itLast) < start + maxDiff) {
            sum += fastHypot((*itLast)[0] - (*itLast)[2], (*itLast)[1] - (*itLast)[3]);
            ++itLast;
        }
        if (sum > maxSum) {
            itBegin = itFirst;
            itEnd = itLast;
            maxSum = sum;
        }

        sum -= fastHypot((*itFirst)[0] - (*itFirst)[2], (*itFirst)[1] - (*itFirst)[3]);
        ++itFirst;
    }

    linesP = {itBegin, itEnd};

    imshow("Transform", dst);
    if (do_imshow) {
        Mat cdstP;
        cvtColor(dst, cdstP, COLOR_GRAY2BGR);

        for (Vec4i& l : linesP) {
            auto color = (min(l[1], l[3]) < 380) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), color, 1, LINE_AA);
        }
        imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    }

    auto reducedLines0 = reduceLines(linesP, 25, 1., 3);

    {
        // find prevailing direction
        std::vector<Point2i> pointCloud;
        for (auto& reduced : reducedLines0) {
            auto centerX = (reduced[0] + reduced[2]) / 2;
            auto centerY = (reduced[1] + reduced[3]) / 2;
            pointCloud.emplace_back(reduced[0] - centerX, reduced[1] - centerY);
            pointCloud.emplace_back(reduced[2] - centerX, reduced[3] - centerY);
        }
        Vec4f lineParams;
        fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);
        const auto tan_phi = lineParams[0] / lineParams[1];

        reducedLines0.erase(std::remove_if(reducedLines0.begin(), reducedLines0.end(),
                                           [tan_phi](const Vec4i& line) {
                                               return fastHypot(line[2] - line[0], line[3] - line[1]) <= 10 ||
                                                      //(gloriousAttempt ? 5 : 10) ||
                                                      fabs(double(line[2] - line[0]) / (line[3] - line[1]) - tan_phi) > 0.07;
                                           }),
                            reducedLines0.end());
    }

    auto reducedLines = reduceLines(reducedLines0, 50, 0.7, 4);

    if (do_imshow) {
        Mat reducedLinesImg0 = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
        RNG rng(215526);
        for (auto& reduced : reducedLines0) {
            auto color = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));
            line(reducedLinesImg0, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), color, 2);
        }
        imshow("Reduced Lines 0", reducedLinesImg0);
    }

    // find prevailing direction
    std::vector<Point2i> pointCloud;
    for (auto& reduced : reducedLines) {
        auto centerX = (reduced[0] + reduced[2]) / 2;
        auto centerY = (reduced[1] + reduced[3]) / 2;
        pointCloud.emplace_back(reduced[0] - centerX, reduced[1] - centerY);
        pointCloud.emplace_back(reduced[2] - centerX, reduced[3] - centerY);
    }
    Vec4f lineParams;
    fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);
    auto cos_phi = lineParams[1];
    auto sin_phi = lineParams[0];
    if (cos_phi < 0) {
        cos_phi = -cos_phi;
        sin_phi = -sin_phi;
    }

    auto sortLam = [cos_phi, sin_phi](const Vec4i& detectedLine) {
        double x = (detectedLine[0] + detectedLine[2]) / 2.;
        double y = (detectedLine[1] + detectedLine[3]) / 2.;
        double x_new = x * cos_phi - y * sin_phi;
        return x_new;
    };

    std::sort(reducedLines.begin(), reducedLines.end(),
              [&sortLam](const Vec4i& l1, const Vec4i& l2) { return sortLam(l1) < sortLam(l2); });

    auto approveLam = [](const Vec4i& line) { return fastHypot(line[2] - line[0], line[3] - line[1]) > 45; };

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

    // Cutting lines
    const auto removeThreshold = 33;
    for (int i = reducedLines.size(); --i >= 0;) {
        auto& line = reducedLines[i];
        double x = (line[0] + line[2]) / 2.;
        double y = CalcPoly(poly, std::clamp(x - WINDOW_DIMENSION_X / 2, double(x_min), double(x_max)) * POLY_COEFF) +
                   WINDOW_DIMENSION_Y / 2;
        if (y < line[1] + removeThreshold ||
            i < reducedLines.size() - 1 && line[3] < reducedLines[i + 1][1]) {  // y > line[3] + tooHighThreshold) {
            reducedLines.erase(reducedLines.begin() + i);
        } else if (y > line[3]) {
            continue;
        } else {
            line[2] = line[0] + double(line[2] - line[0]) / (line[3] - line[1]) * (y - line[1]);
            line[3] = y;
        }
    }

    if (do_imshow) {
        Mat reducedLinesImg = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
        RNG rng(215526);
        for (auto& reduced : reducedLines) {
            auto color = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));
            line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), color, 2);
        }
        imshow("Reduced Lines", reducedLinesImg);
    }

    //////////////////////////////////////////////////////////////////////////

    // turtle stuff

    std::deque<std::pair<Point, Point>> turtleLines;

    const double correction_coeff = 0.2;

    for (int i = goodkeypoints.size(); --i >= 0;) {
        auto& kp = goodkeypoints[i];
        cv::Point pos(kp.pt);
        auto start = FindPath(skeleton, pos);

        if (start.y > pos.y - 10) {
            goodkeypoints.erase(goodkeypoints.begin() + i);
            continue;
        }

        pos.x += kp.size * sin_phi * correction_coeff;
        pos.y += kp.size * cos_phi * correction_coeff;

        turtleLines.emplace_front(start, pos);
    }

    for (int i = goodkeypoints.size(); --i >= 0;) {
        bool erase = false;
        for (int j = goodkeypoints.size(); --j >= 0;) {
            if (i == j) {
                continue;
            }
            if (turtleLines[i].first == turtleLines[j].first && abs(turtleLines[i].second.x - turtleLines[j].second.x) < 10) {
                double y_i =
                    CalcPoly(poly, std::clamp(turtleLines[i].second.x - WINDOW_DIMENSION_X / 2, x_min, x_max) * POLY_COEFF) +
                    WINDOW_DIMENSION_Y / 2;
                double y_j =
                    CalcPoly(poly, std::clamp(turtleLines[j].second.x - WINDOW_DIMENSION_X / 2, x_min, x_max) * POLY_COEFF) +
                    WINDOW_DIMENSION_Y / 2;
                if (abs(y_j - turtleLines[j].second.y) < abs(y_i - turtleLines[i].second.y)) {
                    erase = true;
                    break;
                }
            }
        }
        if (erase) {
            goodkeypoints.erase(goodkeypoints.begin() + i);
            turtleLines.erase(turtleLines.begin() + i);
        }
    }

    // if (do_imshow) {
    //    cv::Mat outSkeleton;
    //    cv::drawKeypoints(skeleton, goodkeypoints, outSkeleton, {0, 255, 0});  // , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //    for (auto& l : turtleLines) {
    //        int radius = 2;
    //        int thickness = -1;
    //        circle(outSkeleton, l.first, radius, {0, 255, 0}, thickness);
    //        line(outSkeleton, l.second, l.first, {0, 255, 0});
    //    }

    //    imshow("outSkeleton", outSkeleton);
    //}

    if (reducedLines.size() < 3) {
        return {};
    }

    {
        // quick'n'dirty fix
        auto sample = reducedLines.back();
        auto step = (sample - reducedLines.front()) / int(reducedLines.size() - 1);

        for (int i = reducedLines.size(); i < 32; ++i) {
            sample += step;
            reducedLines.push_back(sample);
        }
    }

    // Cutting lines once more
    for (int i = reducedLines.size(); --i >= 0;) {
        const auto removeThreshold = 5;
        auto& line = reducedLines[i];
        double x = (line[0] + line[2]) / 2.;
        double y = CalcPoly(poly, std::clamp(x - WINDOW_DIMENSION_X / 2, double(x_min), double(x_max)) * POLY_COEFF) +
                   WINDOW_DIMENSION_Y / 2;
        if (y > line[3]) {
            continue;
        }
        if (y > line[1] + removeThreshold) {
            line[2] = line[0] + double(line[2] - line[0]) / (line[3] - line[1]) * (y - line[1]);
            line[3] = y;
        }
    }

    // Trying to apply turtle stuff
    for (auto& line : reducedLines) {
        for (int shift : {0, 1, -1, 2, -2}) {
            cv::Point pos(line[0] + shift, line[1]);
            doFindPath(skeleton, pos, pos, 0, 0);
            if (pos.y != line[1]) {
                line[0] = pos.x;
                line[1] = pos.y;
                break;
            }
        }
    }

    // Merge turtleLines and reducedLines

    std::vector<std::tuple<double, double, double, double, double>> result;

    for (int i = 0; i < int(reducedLines.size()) - 1; ++i) {
        auto& first = reducedLines[i];
        auto& second = reducedLines[i + 1];

        auto y_first = first[1];
        auto x_first = first[0];

        auto y_second = (first[3] + second[3]) / 2.;
        auto x_second = (first[2] + second[2]) / 2.;

        for (auto& l : turtleLines) {
            if (l.second.x > first[2] && l.second.x < second[2]) {
                y_first = l.first.y;
                x_first = l.first.x;
                y_second = l.second.y;
                x_second = l.second.x;

                break;
            }
        }

        const auto y_first_rotated = first[0] * sin_phi + first[1] * cos_phi;
        const auto y_second_rotated = x_second * sin_phi + y_second * cos_phi;

        const auto diff = (y_second_rotated - y_first_rotated) * originalSize.height / IMAGE_DIMENSION;

        auto convertX = [&originalSize, mirrorX](int x) {
            int result = x * originalSize.width / IMAGE_DIMENSION;
            if (mirrorX) {
                result = originalSize.width - 1 - result;
            }
            return result;
        };
        auto convertY = [&originalSize, mirrorY](int y) {
            int result = y * originalSize.height / IMAGE_DIMENSION;
            if (mirrorY) {
                result = originalSize.height - 1 - result;
            }
            return result;
        };

        result.emplace_back(convertX(x_first), convertY(y_first), convertX(x_second), convertY(y_second), diff);
    }

    if (result.empty()) {
        return {};
    }

    return result;
}

cv::Mat imaging::filter(const cv::Mat& src) {

    cv::Mat img;
    src.convertTo(img, CV_32F);

    if ((src.type() & CV_MAT_DEPTH_MASK) == CV_16U) {
        img /= 256.;
    }

    const auto kernel_size = 3;
    cv::Mat dst;
    GaussianBlur(img, dst, cv::Size(kernel_size, kernel_size), 0, 0);

    dst += 1.;
    cv::log(dst, dst);

    cv::Mat stripeless;
    GaussianBlur(dst, stripeless, cv::Size((63 * src.cols / IMAGE_DIMENSION) | 1, 1), 0, 0);

    cv::Mat funcFloat = dst - stripeless;
    normalize(funcFloat, funcFloat, -64, 255 + 64, cv::NORM_MINMAX);

    funcFloat += 0x23 - cv::mean(funcFloat)[0];

    return funcFloat;
}

std::vector<std::vector<cv::Point>> imaging::transmogrify(const cv::Mat& src) {

    if (src.empty()) {
        throw std::runtime_error("Error opening image");
    }

    const Size originalSize(src.cols, src.rows);

    cv::Mat img;
    src.convertTo(img, CV_32F);

    cv::resize(img, img, cv::Size(IMAGE_DIMENSION, IMAGE_DIMENSION), 0, 0, cv::INTER_LANCZOS4);

    Scalar m = cv::mean(img);
    // std::cout << "*** Mean value: " << m << '\n';

    auto ms = moments(img);
    const double base = ms.m00 * (IMAGE_DIMENSION - 1.) / 2;
    const bool mirrorX = ms.m10 > base;
    const bool mirrorY = ms.m01 > base;

    if (mirrorX) {
        flip(img, img, 1);
    }
    if (mirrorY) {
        flip(img, img, 0);
    }

    const auto kernel_size = 3;
    cv::Mat dst;
    GaussianBlur(img, dst, cv::Size(kernel_size, kernel_size), 0, 0);

    // median stuff

    const float thr = FindThresholdIIntensity(dst);

    // mask
    Mat mask = dst < thr;

    auto funcFloat = filter(img);

    //////////////////////////////////////////////////////////////////////////////

    cv::Mat imgCoherency;
    cv::Mat imgOrientation;
    calcGST(funcFloat, imgCoherency, imgOrientation);

    cv::Mat imgOrientationBin;
    inRange(imgOrientation, cv::Scalar(CV_PI / 2 - 0.2), cv::Scalar(CV_PI / 2 + 0.2), imgOrientationBin);

    cv::Mat imgCoherencyBin = imgCoherency > 0.28;

    auto [equilavenceClassesCount, ptSet, labels] = groupByFourier(img, mask & imgCoherencyBin & imgOrientationBin);

    std::vector<decltype(ptSet)> ptSets(equilavenceClassesCount);
    for (int i = 0; i < ptSet.size(); ++i) {
        ptSets[labels[i]].push_back(ptSet[i]);
    }

    ptSets.erase(std::remove_if(ptSets.begin(), ptSets.end(), [](const auto& v) { return v.size() < n_samples; }), ptSets.end());

    auto convertX = [&originalSize, mirrorX](int x) {
        int result = x * originalSize.width / IMAGE_DIMENSION;
        if (mirrorX) {
            result = originalSize.width - 1 - result;
        }
        return result;
    };
    auto convertY = [&originalSize, mirrorY](int y) {
        int result = y * originalSize.height / IMAGE_DIMENSION;
        if (mirrorY) {
            result = originalSize.height - 1 - result;
        }
        return result;
    };

    std::vector<std::vector<cv::Point>> total_points_fitted;

    for (auto& ptSet : ptSets) {
        cv::Mat poly = polySolve(ptSet);

        int x_min = INT_MAX;
        int x_max = INT_MIN;
        // limits
        for (auto& v : ptSet) {
            auto y = CalcPoly(poly, v.x * POLY_COEFF);
            if (fabs(y - v.y) < 15) {
                x_min = std::min(x_min, v.x);
                x_max = std::max(x_max, v.x);
            }
        }

        std::vector<cv::Point> points_fitted;
        for (int x = x_min; x <= x_max; x++) {
            double y = CalcPoly(poly, x * POLY_COEFF);
            points_fitted.emplace_back(convertX(x + WINDOW_DIMENSION_X / 2), convertY(y + WINDOW_DIMENSION_Y / 2));
        }

        total_points_fitted.push_back(std::move(points_fitted));
    }

    return total_points_fitted;
}
