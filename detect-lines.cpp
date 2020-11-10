#include "detect-lines.h"

#include "known-good.h"

#include "tswdft2d.h"


#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/photo.hpp>

#include <opencv2/plot.hpp>

#include <opencv2/xfeatures2d.hpp>

#include <opencv2/ximgproc.hpp>

#include <ceres/ceres.h>

#include "nanoflann.hpp"

#include <iostream>
#include <map>

#include <random>

#include <array>

#include <deque>

#include <functional>


using namespace cv;

using namespace cv::ximgproc;

namespace {

using namespace cv;


void doFindPath(const cv::Mat& mat, const cv::Point& pt, cv::Point& final, int vertical, float cumulativeAngle)
{
    if (pt.x < 0 || pt.x >= mat.cols || pt.y < 0 || pt.y >= mat.rows)
        return;

    int dist = pt.y - final.y;

    if (abs(vertical) > ((dist > 5)? 1 : 5))
        return;

    if (fabs(cumulativeAngle) > ((dist > 5) ? 1.8 : 10.))
        return;

    if (mat.at<uchar>(pt) == 0)
        return;

    if (final.y > pt.y)
        final = pt;

    cumulativeAngle *= 0.8;

    doFindPath(mat, Point(pt.x, pt.y - 1), final, 0, cumulativeAngle);
    doFindPath(mat, Point(pt.x + 1, pt.y - 1), final, 0, cumulativeAngle + 0.5);
    doFindPath(mat, Point(pt.x - 1, pt.y - 1), final, 0, cumulativeAngle - 0.5);
    if (vertical >= 0)
        doFindPath(mat, Point(pt.x + 1, pt.y), final, vertical + 1, cumulativeAngle + 1);
    if (vertical <= 0)
        doFindPath(mat, Point(pt.x - 1, pt.y), final, vertical - 1, cumulativeAngle - 1);
}

cv::Point FindPath(const cv::Mat& mat, const cv::Point& start)
{
    cv::Point pos = start;

    while (pos.x >= 0 && (mat.at<uchar>(pos) == 0 || (doFindPath(mat, pos, pos, 0, 0), pos.y == start.y)))
        --pos.x;

    if (pos.x < 0)
        return start;

    //doFindPath(mat, pos, pos, 0, 0);

    return pos;
}


//////////////////////////////////////////////////////////////////////////////

auto extendedLine(const Vec4i& line, double d, double max_coeff) {
    const auto length = hypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = std::min(d / length, max_coeff);
    double xd = (line[2] - line[0]) * coeff;
    double yd = (line[3] - line[1]) * coeff;
    return Vec4f(line[0] - xd, line[1] - yd, line[2] + xd, line[3] + yd);
}

std::array<Point2f, 4> boundingRectangleContour(const Vec4i& line, float d) {
    // finds coordinates of perpendicular lines with length d in both line points
    const auto length = hypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = d / length;

    // dx:  -dy
    // dy:  dx
    double yd = (line[2] - line[0]) * coeff;
    double xd = -(line[3] - line[1]) * coeff;

    return  {
        Point2f(line[0]-xd, line[1]-yd),
        Point2f(line[0]+xd, line[1]+yd),
        Point2f(line[2]+xd, line[3]+yd),
        Point2f(line[2]-xd, line[3]-yd)
    };
}

bool extendedBoundingRectangleLineEquivalence(const Vec4i& l1, const Vec4i& l2,
    float extensionLength, float extensionLengthMaxFraction,
    float boundingRectangleThickness) {

    const auto el1 = extendedLine(l1, extensionLength, extensionLengthMaxFraction);
    const auto el2 = extendedLine(l2, extensionLength, extensionLengthMaxFraction);

    // calculate window around extended line
    // at least one point needs to inside extended bounding rectangle of other line,
    const auto lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
    return
        pointPolygonTest(lineBoundingContour, { el2[0], el2[1] }, false) >= 0 ||
        pointPolygonTest(lineBoundingContour, { el2[2], el2[3] }, false) >= 0 ||

        pointPolygonTest(lineBoundingContour, Point2f(l2[0], l2[1]), false) >= 0 ||
        pointPolygonTest(lineBoundingContour, Point2f(l2[2], l2[3]), false) >= 0;
}

Vec4i HandlePointCloud(const std::vector<Point2i>& pointCloud) {
    //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
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
        pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{},
                [](std::vector<Vec4i> target, const std::vector<Point2i>& pointCloud) {
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






//////////////////////////////////////////////////////////////////////////////

void calcGST(const cv::Mat& inputImg, cv::Mat& imgCoherencyOut, cv::Mat& imgOrientationOut, int w = 52)
{
    using namespace cv;

    Mat img;
    inputImg.convertTo(img, CV_32F);
    // GST components calculation (start)
    // J =  (J11 J12; J12 J22) - GST
    Mat imgDiffX, imgDiffY, imgDiffXY;
    Sobel(img, imgDiffX, CV_32F, 1, 0, 3);
    Sobel(img, imgDiffY, CV_32F, 0, 1, 3);
    multiply(imgDiffX, imgDiffY, imgDiffXY);
    Mat imgDiffXX, imgDiffYY;
    multiply(imgDiffX, imgDiffX, imgDiffXX);
    multiply(imgDiffY, imgDiffY, imgDiffYY);
    Mat J11, J22, J12;      // J11, J22 and J12 are GST components
    boxFilter(imgDiffXX, J11, CV_32F, Size(w, w));
    boxFilter(imgDiffYY, J22, CV_32F, Size(w, w));
    boxFilter(imgDiffXY, J12, CV_32F, Size(w, w));
    // GST components calculation (stop)
    // eigenvalue calculation (start)
    // lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
    // lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
    Mat tmp1, tmp2, tmp3, tmp4;
    tmp1 = J11 + J22;
    tmp2 = J11 - J22;
    multiply(tmp2, tmp2, tmp2);
    multiply(J12, J12, tmp3);
    sqrt(tmp2 + 4.0 * tmp3, tmp4);
    Mat lambda1, lambda2;
    lambda1 = tmp1 + tmp4;
    lambda1 = 0.5*lambda1;      // biggest eigenvalue
    lambda2 = tmp1 - tmp4;
    lambda2 = 0.5*lambda2;      // smallest eigenvalue
    // eigenvalue calculation (stop)
    // Coherency calculation (start)
    // Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    // Coherency is anisotropy degree (consistency of local orientation)
    divide(lambda1 - lambda2, lambda1 + lambda2, imgCoherencyOut);
    // Coherency calculation (stop)
    // orientation angle calculation (start)
    // tan(2*Alpha) = 2*J12/(J22 - J11)
    // Alpha = 0.5 atan2(2*J12/(J22 - J11))
    phase(J22 - J11, 2.0*J12, imgOrientationOut, false);
    imgOrientationOut = 0.5*imgOrientationOut;
    // orientation angle calculation (stop)
}


const int IMAGE_DIMENSION = 800;
//const int IMAGE_DIMENSION = 512;

enum { WINDOW_DIMENSION_X = 64 };
enum { WINDOW_DIMENSION_Y = 1 };

const auto visualizationRows = IMAGE_DIMENSION - WINDOW_DIMENSION_Y + 1;
const auto visualizationCols = IMAGE_DIMENSION - WINDOW_DIMENSION_X + 1;


void DemoShow(const cv::Mat& src, const char* caption)
{
    auto copy = src.clone();
    resize(copy, copy, cv::Size(256, 256));
    imshow(caption, copy);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    using namespace cv;

    const auto& transformed = *static_cast<std::vector<std::complex<float>>*>(userdata);

    //enum { WINDOW_DIMENSION = 32 };

    enum { HALF_SIZE_X = WINDOW_DIMENSION_X / 2 };
    enum { HALF_SIZE_Y = WINDOW_DIMENSION_Y / 2 };

    if (event == cv::EVENT_MOUSEMOVE)
    {
        const auto xx = std::min(std::max(x - HALF_SIZE_X, 0), IMAGE_DIMENSION - WINDOW_DIMENSION_X);
        const auto yy = std::min(std::max(y - HALF_SIZE_Y, 0), IMAGE_DIMENSION - WINDOW_DIMENSION_Y);

        const auto sourceOffset = yy * visualizationCols + xx;


        std::map<float, float> ordered;

        for (int j = 1; j < WINDOW_DIMENSION_X/* * WINDOW_DIMENSION*/; ++j)
        {
            if (j / WINDOW_DIMENSION_X > WINDOW_DIMENSION_Y / 2 || j % WINDOW_DIMENSION_X > WINDOW_DIMENSION_X / 2)
                continue;

            const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X + j]);
            const auto freq = hypot(j / WINDOW_DIMENSION_X, j % WINDOW_DIMENSION_X);
            if (freq > 2)
                ordered[freq] = std::max(ordered[freq], amplitude);
        }

        Mat data_x; //(1, 51, CV_64F);
        Mat data_y; //(1, 51, CV_64F);

        data_x.push_back(0.);
        data_y.push_back(0.);

        for (auto& v : ordered)
        {
            data_x.push_back(double(v.first));
            data_y.push_back(double(v.second));
        }

        //cv::normalize(data_y, data_y);


        Mat plot_result;

        Ptr<plot::Plot2d> plot = plot::Plot2d::create(data_x, data_y);
        //plot->render(plot_result);
        //imshow("The plot", plot_result);

        plot->setShowText(false);
        plot->setShowGrid(false);
        plot->setPlotBackgroundColor(Scalar(255, 200, 200));
        plot->setPlotLineColor(Scalar(255, 0, 0));
        plot->setPlotLineWidth(2);
        plot->setInvertOrientation(true);
        plot->render(plot_result);

        imshow("The plot", plot_result);

        cv::Mat magI(WINDOW_DIMENSION_Y, WINDOW_DIMENSION_X, CV_32FC1);
        for (int j = 1; j < WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X; ++j)
        {
            const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X + j]);
            magI.at<float>(j / WINDOW_DIMENSION_X, j % WINDOW_DIMENSION_X) = amplitude;
        }

        magI += Scalar::all(1);                    // switch to logarithmic scale
        log(magI, magI);

        normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a

        auto magIcopy = magI.clone();
        cv::resize(magIcopy, magIcopy, cv::Size(512, 512));

        imshow("original spectrum magnitude", magIcopy);

        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = magI.cols / 2;
        int cy = magI.rows / 2;
        Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
        Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
        Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

        //DemoShow(q0, "q0");
        //DemoShow(q1, "q1");
        //DemoShow(q2, "q2");
        //DemoShow(q3, "q3");

        Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
                                                // viewable image form (float between values 0 and 1).
        //imshow("Input Image", I);    // Show the result

        cv::resize(magI, magI, cv::Size(512, 512));

        imshow("spectrum magnitude", magI);



        //cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

        //cv::Rect roi(cv::Point(std::max(x - HALF_SIZE, 0), std::max(y - HALF_SIZE, 0)),
        //    cv::Point(std::min(x + HALF_SIZE - 1, IMAGE_DIMENSION), std::min(y + HALF_SIZE - 1, IMAGE_DIMENSION)));


        //displayFourier((*m)(roi));

        //auto copy = (*m)(roi).clone();
        //auto line = copy.reshape(0, 1);
        //const auto mean = cv::mean(line);
        //line -= mean;
        //cv::normalize(line, line);

        //displayFourier(copy);
    }
}

//////////////////////////////////////////////////////////////////////////////

const double POLY_COEFF = 0.001;

bool polynomial_curve_fit(const std::vector<cv::Point2d>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();

    //Construct matrix X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                    std::pow(key_point[k].x, i + j);
            }
        }
    }

    //Construct matrix Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //Solve matrix A
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return true;
}


// Generate a uniform distribution of the number between[0, 1]
double uniformRandom(void)
{
    return (double)rand() / (double)RAND_MAX;
}

/*
// Fit the line according to the point set ax + by + c = 0, res is the residual
void calcLinePara(const std::vector<cv::Point2d>& pts, double &a, double &b, double &c, double &res)
{
    res = 0;
    cv::Vec4f line;
    std::vector<cv::Point2f> ptsF;
    for (unsigned int i = 0; i < pts.size(); i++)
        ptsF.push_back(pts[i]);

    cv::fitLine(ptsF, line, cv::DIST_L2, 0, 1e-2, 1e-2);
    a = line[1];
    b = -line[0];
    c = line[0] * line[3] - line[1] * line[2];

    for (unsigned int i = 0; i < pts.size(); i++)
    {
        double resid_ = fabs(pts[i].x * a + pts[i].y * b + c);
        res += resid_;
    }
    res /= pts.size();
}
*/

// Get a straight line fitting sample, that is, randomly select 2 points on the line sampling point set
#if 0
bool getSample(const std::vector<int>& set, std::vector<int> &sset, int num)
{
    if (set.size() <= num)
        return false;

    std::map<int, int> displaced;

    sset.resize(num);

    std::default_random_engine dre;
    std::uniform_int_distribution<int> di(0, set.size() - 1);

    for (int i = 0; i < num; ++i)
    {
        int idx = di(dre);
        int v;
        if (idx == i)
            v = i;
        else
        {
            auto it = displaced.find(idx);
            if (it != displaced.end())
            {
                v = it->second;
                it->second = i;
            }
            else
            {
                v = idx;
                displaced[idx] = i;
            }
        }

        sset[i] = v;
    }

    return true;

    //int i[2];
    //sset.resize(n);
    //if (set.size() > 2)
    //{
    //    do
    //    {
    //        for (int n = 0; n < 2; n++)
    //            i[n] = int(uniformRandom() * (set.size() - 1));
    //    } while (!(i[1] != i[0]));
    //    for (int n = 0; n < 2; n++)
    //    {
    //        sset.push_back(i[n]);
    //    }
    //}
    //else
    //{
    //    return false;
    //}
    //return true;
}
#endif

//The position of two random points in the line sample cannot be too close
bool verifyComposition(const std::vector<cv::Point2d>& pts)
{
    //cv::Point2d pt1 = pts[0];
    //cv::Point2d pt2 = pts[1];
    //if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
    //    return false;

    for (int i = 1; i < pts.size(); ++i)
        for (int j = 0; j < i; ++j)
        {
            cv::Point2d pt1 = pts[j];
            cv::Point2d pt2 = pts[i];
            if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
                return false;
        }

    return true;
}


//RANSAC straight line fitting
void fitLineRANSAC(const std::vector<cv::Point>& ptSet,
    //double &a, double &b, double &c,
    cv::Mat& a, int n_samples,
    std::vector<bool> &inlierFlag)
{
    //double residual_error = 2.99; // inner point threshold
    const double residual_error = 10; // inner point threshold

    bool stop_loop = false;
    int maximum = 0; //maximum number of points

    //final inner point identifier and its residual
    inlierFlag = std::vector<bool>(ptSet.size(), false);
    std::vector<double> resids_;// (ptSet.size(), 3);
    int sample_count = 0;
    int N = 100000;

    //double res = 0;

    // RANSAC
    srand((unsigned int)time(NULL)); //Set random number seed
    std::vector<int> ptsID;
    for (unsigned int i = 0; i < ptSet.size(); i++)
        ptsID.push_back(i);

    //enum { n_samples  = 8 };

    std::vector<int> ptss(n_samples);

    std::default_random_engine dre;

    while (N > sample_count && !stop_loop)
    {
        cv::Mat res;

        std::vector<bool> inlierstemp;
        std::vector<double> residualstemp;
        //std::vector<int> ptss;
        int inlier_count = 0;

        //random sampling - n_samples points
        for (int j = 0; j < n_samples; ++j)
            ptss[j] = j;

        std::map<int, int> displaced;

        // Fisher-Yates shuffle Algorithm
        for (int j = 0; j < n_samples; ++j)
        {
            std::uniform_int_distribution<int> di(j, ptsID.size() - 1);
            int idx = di(dre);

            if (idx != j)
            {
                int& to_exchange = (idx < n_samples) ? ptss[idx] : displaced.try_emplace(idx, idx).first->second;
                std::swap(ptss[j], to_exchange);
            }
        }


        //if (!getSample(ptsID, ptss, 3))
        //{
        //    stop_loop = true;
        //    continue;
        //}

        std::vector<cv::Point2d> pt_sam;
        for (int i = 0; i < n_samples; ++i)
            pt_sam.push_back(ptSet[ptss[i]]);

        //pt_sam.push_back(ptSet[ptss[0]]);
        //pt_sam.push_back(ptSet[ptss[1]]);

        if (!verifyComposition(pt_sam))
        {
            ++sample_count;
            continue;
        }

        // Calculate the line equation
            //calcLinePara(pt_sam, a, b, c, res);
        polynomial_curve_fit(pt_sam, n_samples - 1, res);
        //Inside point test
        for (unsigned int i = 0; i < ptSet.size(); i++)
        {
            cv::Point2d pt = ptSet[i];
            auto x = ptSet[i].x;
            //double resid_ = fabs(pt.x * a + pt.y * b + c);

            //double y = res.at<double>(0, 0) + res.at<double>(1, 0) * x +
            //    res.at<double>(2, 0)*std::pow(x, 2) + res.at<double>(3, 0)*std::pow(x, 3);

            double y = res.at<double>(0, 0) + res.at<double>(1, 0) * x;
            for (int i = 2; i < n_samples; ++i)
                y += res.at<double>(i, 0) * std::pow(x, i);

            double resid_ = fabs(ptSet[i].y - y);

            residualstemp.push_back(resid_);
            inlierstemp.push_back(false);
            if (resid_ < residual_error)
            {
                ++inlier_count;
                inlierstemp[i] = true;
            }
        }
        // find the best fit straight line
        if (inlier_count >= maximum)
        {
            maximum = inlier_count;
            resids_ = residualstemp;
            inlierFlag = inlierstemp;
        }
        // Update the number of RANSAC iterations, as well as the probability of interior points
        if (inlier_count == 0)
        {
            N = 500;
        }
        else
        {
            double epsilon = 1.0 - double(inlier_count) / (double)ptSet.size(); // wild value point scale
            double p = 0.99; //the probability of having 1 good sample in all samples
            double s = 2.0;
            N = int(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
        }
        ++sample_count;
    }

    // Use all the interior points to re - fit the line
    std::vector<cv::Point2d> pset;
    for (unsigned int i = 0; i < ptSet.size(); i++)
    {
        if (inlierFlag[i])
            pset.push_back(ptSet[i]);
    }

    //calcLinePara(pset, a, b, c, res);
    polynomial_curve_fit(pset, n_samples - 1, a);
}

//////////////////////////////////////////////////////////////////////////////

double CalcPoly(const cv::Mat& X, double x)
{
    double result = X.at<double>(0, 0);
    double v = 1.;
    for (int i = 1; i < X.rows; ++i)
    {
        v *= x;
        result += X.at<double>(i, 0) * v;
    }
    return result;
}

void fitLineRANSAC2(const std::vector<cv::Point>& vals, cv::Mat& a, int n_samples, std::vector<bool> &inlierFlag, double noise_sigma = 5.)
{
    //int n_data = vals.size();
    int N = 5000;	//iterations
    double T = 3 * noise_sigma;   // residual threshold

    //int n_sample = 3;

    //int max_cnt = 0;

    double max_weight = 0.;

    cv::Mat best_model(n_samples, 1, CV_64FC1);

    std::default_random_engine dre;

    std::vector<int> k(n_samples);

    for (int n = 0; n < N; n++)
    {
        //random sampling - n_samples points
        for (int j = 0; j < n_samples; ++j)
            k[j] = j;

        std::map<int, int> displaced;

        // Fisher-Yates shuffle Algorithm
        for (int j = 0; j < n_samples; ++j)
        {
            std::uniform_int_distribution<int> di(j, vals.size() - 1);
            int idx = di(dre);

            if (idx != j)
            {
                int& to_exchange = (idx < n_samples) ? k[idx] : displaced.try_emplace(idx, idx).first->second;
                std::swap(k[j], to_exchange);
            }
        }

        //printf("random sample : %d %d %d\n", k[0], k[1], k[2]);

        //model estimation
        cv::Mat AA(n_samples, n_samples, CV_64FC1);
        cv::Mat BB(n_samples, 1, CV_64FC1);
        for (int i = 0; i < n_samples; i++)
        {
            AA.at<double>(i, 0) = 1.;
            double v = 1.;
            for (int j = 1; j < n_samples; ++j)
            {
                v *= vals[k[i]].x * POLY_COEFF;
                AA.at<double>(i, j) = v;
            }

            BB.at<double>(i, 0) = vals[k[i]].y;
        }

        cv::Mat AA_pinv(n_samples, n_samples, CV_64FC1);
        invert(AA, AA_pinv, cv::DECOMP_SVD);

        cv::Mat X = AA_pinv * BB;

        //evaluation
        //int cnt = 0;
        std::map<int, double> bestValues;
        double weight = 0.;
        for (const auto& v : vals)
        {
            const double arg = std::abs(v.y - CalcPoly(X, v.x * POLY_COEFF));
            const double data = exp(-arg * arg / (2 * noise_sigma * noise_sigma));

            auto& val = bestValues[v.x];
            if (data > val)
            {
                weight += data - val;
                val = data;
            }

            //if (data < T)
            //{
            //    cnt++;
            //}
        }

        //if (cnt > max_cnt)
        if (weight > max_weight)
        {
            best_model = X;
            max_weight = weight;
        }
    }

    //------------------------------------------------------------------- optional LS fitting
    inlierFlag = std::vector<bool>(vals.size(), false);
    std::vector<int> vec_index;
    for (int i = 0; i < vals.size(); i++)
    {
        const auto& v = vals[i];
        double data = std::abs(v.y - CalcPoly(best_model, v.x * POLY_COEFF));
        if (data < T)
        {
            inlierFlag[i] = true;
            vec_index.push_back(i);
        }
    }

    cv::Mat A2(vec_index.size(), n_samples, CV_64FC1);
    cv::Mat B2(vec_index.size(), 1, CV_64FC1);

    for (int i = 0; i < vec_index.size(); i++)
    {
        A2.at<double>(i, 0) = 1.;
        double v = 1.;
        for (int j = 1; j < n_samples; ++j)
        {
            v *= vals[vec_index[i]].x * POLY_COEFF;
            A2.at<double>(i, j) = v;
        }


        B2.at<double>(i, 0) = vals[vec_index[i]].y;
    }

    cv::Mat A2_pinv(n_samples, vec_index.size(), CV_64FC1);
    invert(A2, A2_pinv, cv::DECOMP_SVD);

    a = A2_pinv * B2;

    //return X;
}

//////////////////////////////////////////////////////////////////////////////

struct PolynomialResidual
{
    PolynomialResidual(double x, double y, int n_samples)
        : x_(x), y_(y), n_samples_(n_samples) {}

    template <typename T>
    bool operator()(T const* const* relative_poses, T* residuals) const {

        T y = *(relative_poses[0]) + *(relative_poses[1]) * x_;
        for (int i = 2; i < n_samples_; ++i)
            y += *(relative_poses[i]) * std::pow(x_, i);

        residuals[0] = T(y_) - y;
        return true;
    }

private:
    // Observations for a sample.
    const double x_;
    const double y_;
    int n_samples_;
};

/*
struct CrappyResidual {
    CrappyResidual(double x, double y)
        : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const m0, const T* const m1, const T* const m2, const T* const m3, const T* const m4, const T* const m5, const T* const m6, const T* const m7, T* residual) const {
        const double relative_poses[] { static_cast<double>(m0[0]), static_cast<double>(m1[0]), static_cast<double>(m2[0]), static_cast<double>(m3[0]),
            static_cast<double>(m4[0]), static_cast<double>(m5[0]), static_cast<double>(m6[0]), static_cast<double>(m7[0]) };
        double y = relative_poses[0] + relative_poses[1] * x_;
        for (int i = 2; i < 8; ++i)
            y += relative_poses[i] * std::pow(x_, i);

        residual[0] = T(y_) - y;

        return true;
    }

private:
    // Observations for a sample.
    const double x_;
    const double y_;
};
*/

//////////////////////////////////////////////////////////////////////////////

class PointsProvider
{
public:
    PointsProvider(const std::vector<cv::Point>& ptSet)
        : ptSet_(ptSet)
    {}

    size_t kdtree_get_point_count() const
    {
        return ptSet_.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        auto& v = ptSet_[idx];
        return dim? v.y : v.x;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }


private:
    const std::vector<cv::Point>& ptSet_;
};

// construct a kd-tree index:
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointsProvider >,
    PointsProvider,
    2 /* dim */
> my_kd_tree_t;


std::vector<std::tuple<double, double, double, double, double>> calculating(
        const std::string& filename, std::function<void(const std::string&, cv::InputArray)> do_imshow)
{
    auto imshow = [do_imshow](const char* winname, InputArray mat) {
        if (do_imshow)
            do_imshow(winname, mat);
    };

    Mat src = imread(filename, IMREAD_GRAYSCALE);

    if (src.empty()) {
        throw std::runtime_error("Error opening image");
    }


    const Size originalSize(src.cols, src.rows);

#if 0
    Mat lowLevelFloat = Mat::zeros(src.size(), CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int v = src.at<uchar>(y, x);
            lowLevelFloat.at<float>(y, x) = std::clamp(v, 0, 1) * 10;
        }
    }


    enum { CONVERTED_SIZE = 800 };

    resize(src, src, Size(CONVERTED_SIZE, CONVERTED_SIZE), 0, 0, INTER_LANCZOS4);


    resize(lowLevelFloat, lowLevelFloat, Size(CONVERTED_SIZE, CONVERTED_SIZE), 0, 0, INTER_LANCZOS4);
    GaussianBlur(lowLevelFloat, lowLevelFloat, Size(9, 35), 0, 0, BORDER_DEFAULT);


    auto ms = moments(src);
    const double base = ms.m00 * (CONVERTED_SIZE - 1.) / 2;
    const bool mirrorX = ms.m10 > base;
    const bool mirrorY = ms.m01 > base;

    if (mirrorX) {
        flip(src, src, 1);
        flip(lowLevelFloat, lowLevelFloat, 1);
    }
    if (mirrorY) {
        flip(src, src, 0);
        flip(lowLevelFloat, lowLevelFloat, 0);
    }


    //////////////////////////////////////////////////////////////////////////

    Mat lowLevel;
    lowLevelFloat.convertTo(lowLevel, CV_8U);

    Mat dstFloat;
    src.convertTo(dstFloat, CV_32F);

    Mat dstFloat2 = dstFloat.clone();

    for (int y = 0; y < dstFloat2.rows; y++) {
        for (int x = 0; x < dstFloat2.cols; x++) {
            const auto threshold = 246;
            int v = dstFloat2.at<float>(y, x);
            if (v > threshold)
                dstFloat2.at<float>(y, x) = v + (v - threshold) * 2;
        }
    }


    dstFloat += lowLevelFloat;
    dstFloat2 += lowLevelFloat;

    src += lowLevel;

    const auto kernel_size = 3;

    GaussianBlur(dstFloat, dstFloat, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);
    GaussianBlur(dstFloat2, dstFloat2, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);


    Mat stripeless;
    GaussianBlur(dstFloat, stripeless, Size(63, 1), 0, 0, BORDER_DEFAULT);

    Mat funcFloat = (dstFloat - stripeless + 32.) * 4.;
    Mat func;
    funcFloat.convertTo(func, CV_8U);


    Mat stripeless2;
    GaussianBlur(dstFloat2, stripeless2, Size(63, 1), 0, 0, BORDER_DEFAULT);

    Mat funcFloat2 = (dstFloat2 - stripeless2 + 32.) * 4.;
    Mat func2;
    funcFloat2.convertTo(func2, CV_8U);

    Mat func4surf = (func - (128 / 8 * 7)) * 8;
    GaussianBlur(func4surf, func4surf, Size(7, 7), 0, 0, BORDER_DEFAULT);


    auto surf = cv::xfeatures2d::SURF::create(5000);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    surf->detectAndCompute(func4surf, cv::noArray(), keypoints, descriptors);

    // http://itnotesblog.ru/note.php?id=271
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(descriptors, GetKnownGood(), matches);

    std::vector< KeyPoint > goodkeypoints;

    for (int i = 0; i < descriptors.rows; i++) {
        if (matches[i].distance < 0.175) {
            goodkeypoints.push_back(keypoints[i]);
        }
    }

    if (goodkeypoints.empty())
        return {};

    std::vector<int> labels;
    int equilavenceClassesCount = cv::partition(goodkeypoints, labels, [](const KeyPoint& k1, const KeyPoint& k2) {
        const auto MAX_DIST = 45;
        if (fabs(k1.pt.x - k2.pt.x) > MAX_DIST || fabs(k1.pt.y - k2.pt.y) > MAX_DIST)
                return false;

        auto[minSize, maxSize] = std::minmax(k1.size, k2.size);
        if (maxSize / minSize > 1.35)
            return false;

        return true;
    });

    std::vector < std::vector<KeyPoint>> outKeypoints(equilavenceClassesCount);
    for (int i = 0; i < goodkeypoints.size(); ++i)
    {
        outKeypoints[labels[i]].push_back(goodkeypoints[i]);
    }

    auto maxSet = std::max_element(outKeypoints.begin(), outKeypoints.end(),
        [](const auto& left, const auto& right) { return left.size() < right.size(); });

    if (maxSet->empty())
        return {};

    for (int i = maxSet->size() - 1; --i >= 0;)
        for (int j = maxSet->size(); --j > i;)
        {
            if (hypot((*maxSet)[i].pt.x - (*maxSet)[j].pt.x, (*maxSet)[i].pt.y - (*maxSet)[j].pt.y) < 5)
            {
                maxSet->erase(maxSet->begin() + j);
            }
        }


    cv::Mat withSurf = func4surf.clone();

    cv::drawKeypoints(withSurf, *maxSet, withSurf, {0, 255, 0});// , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //imshow("withSurf", withSurf);

    Mat thresh = func2.clone();
    thresh += 127;
    //imshow("thresh0", thresh);
    adaptiveThreshold(thresh, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1.0);

    //imshow("thresh", thresh);

    Mat erodeDilate = thresh.clone();
    {
        Mat erodeStructure = getStructuringElement(MORPH_RECT, Size(3, 5));
        erode(erodeDilate, erodeDilate, erodeStructure);
        Mat dilateStructure = getStructuringElement(MORPH_RECT, Size(9, 5));
        dilate(erodeDilate, erodeDilate, dilateStructure);
    }

    //imshow("erodeDilate", erodeDilate);

    Mat skeleton;
    thinning(thresh, skeleton);

    cv::Mat outSkeleton;
    cv::drawKeypoints(skeleton, *maxSet, outSkeleton, { 0, 255, 0 });// , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


    cv::Mat imgOrientation;
    calcGST(func2, imgOrientation);

    std::vector<float> orientations;
    for (auto& kp : *maxSet)
    {
        cv::Point pos(kp.pt);
        auto v = imgOrientation.at<float>(pos);
        orientations.push_back(v);
    }

    const auto mid = orientations.size() / 2;
    std::nth_element(orientations.begin(), orientations.begin() + mid, orientations.end());
    const auto medianAngle = orientations[mid];

    auto cos_phi = sin(medianAngle);
    auto sin_phi = -cos(medianAngle);

    // sort
    auto sortLam = [cos_phi, sin_phi](const KeyPoint& kp) {
        double x_new = kp.pt.x * cos_phi - kp.pt.y * sin_phi;
        return x_new;
    };

    std::sort(maxSet->begin(), maxSet->end(), [&sortLam](const KeyPoint& kp1, const KeyPoint& kp2) {
        return sortLam(kp1) < sortLam(kp2);
    });

    double avgDist = 0;
    {
        auto it = maxSet->begin();
        double distSum = 0;
        double prevX = sortLam(*it);
        while (++it != maxSet->end())
        {
            double x = sortLam(*it);
            distSum += x - prevX;
            prevX = x;
        }

        avgDist = distSum / (maxSet->size() - 1);
    }

    const auto righLong = FindRightPath(skeleton, cv::Point(maxSet->rbegin()->pt));

    cv::Point rightLineStart(righLong);
    rightLineStart.x += avgDist * 3 / 4;
    rightLineStart.y += avgDist / 2;

    cv::Point rightLineEnd;
    rightLineEnd.y = erodeDilate.rows - 1;
    rightLineEnd.x = rightLineStart.x + (rightLineEnd.y - rightLineStart.y) * sin_phi / cos_phi;

    auto rightShort = rightLineStart;

    LineIterator it(erodeDilate, rightLineStart, rightLineEnd);
    for (int i = 0; i < it.count; i++, ++it)
    {
        auto val = erodeDilate.at<uchar>(it.pos());
        if (val != 0)
        {
            rightShort = it.pos();
            break;
        }
    }


    std::vector<std::pair<Point, Point>> reducedLines;

    const double correction_coeff = 0.2;

    for (auto& kp : *maxSet)
    {
        cv::Point pos(kp.pt);
        auto start = FindPath(skeleton, pos);

        pos.x += kp.size * sin_phi * correction_coeff;
        pos.y += kp.size * cos_phi * correction_coeff;

        reducedLines.emplace_back(start, pos);
    }

    reducedLines.emplace_back(righLong, rightShort);


    //////////////////////////////////////////////////////////////////////////



    std::vector<std::tuple<double, double, double, double, double>> result;

    for (auto& line : reducedLines) {
        const auto y_first = line.first.x * sin_phi + line.first.y * cos_phi;
        const auto y_second = line.second.x * sin_phi + line.second.y * cos_phi;

        //const auto x_first = first[0] * cos_phi - first[1] * sin_phi;
        //const auto x_second = second[0] * cos_phi - second[1] * sin_phi;

        const auto diff = (y_second - y_first) * originalSize.height / CONVERTED_SIZE;

        auto convertX = [&originalSize, mirrorX](int x) {
            int result = x * originalSize.width / CONVERTED_SIZE;
            if (mirrorX)
                result = originalSize.width - 1 - result;
            return result;
        };
        auto convertY = [&originalSize, mirrorY](int y) {
            int result = y * originalSize.height / CONVERTED_SIZE;
            if (mirrorY)
                result = originalSize.height - 1 - result;
            return result;
        };

        result.emplace_back(
                convertX(line.first.x),
                convertY(line.first.y),
                convertX(line.second.x),
                convertY(line.second.y),
                diff);
    }
#endif

    cv::Mat img;
    src.convertTo(img, CV_32F);

    //for (int y = 0; y < img.rows; y++) {
    //    for (int x = 0; x < img.cols; x++) {
    //        const auto threshold = 252;
    //        int v = img.at<float>(y, x);
    //        if (v > threshold)
    //            img.at<float>(y, x) = v + (v - threshold) * 16;
    //    }
    //}


    cv::resize(img, img, cv::Size(IMAGE_DIMENSION, IMAGE_DIMENSION), 0, 0, cv::INTER_LANCZOS4);

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



    //cv::Mat special(IMAGE_DIMENSION, IMAGE_DIMENSION, CV_8UC1);
    //for (int y = 0; y < img.rows; y++) {
    //    for (int x = 0; x < img.cols; x++) {
    //        //const auto threshold = 230;
    //        int v = img.at<float>(y, x);

    //        special.at<uchar>(y, x) = (v < 254 && v > 200)? 255 : 0;
    //    }
    //}
    //imshow("special", special);



    const auto kernel_size = 3;
    cv::Mat dst;
    cv::GaussianBlur(img, dst, cv::Size(kernel_size, kernel_size), 0, 0, cv::BORDER_DEFAULT);
    //const auto filtered = dst.clone();

    Mat background;
    GaussianBlur(img, background, Size(63, 63), 0, 0, BORDER_DEFAULT);
    //background -= 1;

    Mat diff = dst < background;

    // mask
    auto thr = cv::mean(dst.row(0))[0];
    std::cout << "Threshold: " << thr << '\n';
    if (thr > 254.)
        thr -= 10;
    else
        thr -= 30;
    Mat mask = dst < thr;
    //dst.convertTo(mask, CV_8U);
    //threshold(dst, mask, 180, 255, THRESH_BINARY_INV);
    int horizontal_size = 40;
    // Create structure element for extracting vertical lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
    // Apply morphology operations
    dilate(mask, mask, horizontalStructure);


    dst += 1.;
    cv::log(dst, dst);

    cv::Mat stripeless;
    GaussianBlur(dst, stripeless, cv::Size(63, 1), 0, 0, cv::BORDER_DEFAULT);

    //cv::Mat funcFloat = (dst - stripeless + 8) * 16;
    cv::Mat funcFloat = dst - stripeless;
    normalize(funcFloat, funcFloat, -64, 255 + 64, cv::NORM_MINMAX);
    cv::Mat func;
    funcFloat.convertTo(func, CV_8U);


    // !!!
    dst = func.clone();



    cv::Mat imgCoherency, imgOrientation;
    calcGST(funcFloat, imgCoherency, imgOrientation);


    cv::GaussianBlur(img, img, cv::Size(1, 33), 0, 0, cv::BORDER_DEFAULT);

    auto transformed = tswdft2d<float>((float*)img.data, WINDOW_DIMENSION_Y, WINDOW_DIMENSION_X, img.rows, img.cols);

    cv::Mat visualization(visualizationRows, visualizationCols, CV_32FC1);
    cv::Mat amplitude(visualizationRows, visualizationCols, CV_32FC1);

    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;

            //float offsets[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];
            //float amplitudes[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];

            std::map<float, float> ordered;

            double v = 0;

            for (int j = 1; j < WINDOW_DIMENSION_X/* * WINDOW_DIMENSION*/; ++j)
            {
                if (j / WINDOW_DIMENSION_X > WINDOW_DIMENSION_Y / 2 || j % WINDOW_DIMENSION_X > WINDOW_DIMENSION_X / 2)
                    continue;

                const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X + j]) / sqrtf(j);
                const auto freq = hypot(j / WINDOW_DIMENSION_X, j % WINDOW_DIMENSION_X);
                if (freq > 2)
                    ordered[freq] = std::max(ordered[freq], amplitude);

                if (j % WINDOW_DIMENSION_X == 0)
                    v += amplitude;
            }

            auto it = ordered.begin();
            auto freq = it->first;
            auto threshold = it->second;

            while (++it != ordered.end())
            {
                if (it->second > threshold)
                {
                    freq = it->first;
                    threshold = it->second;
                }
                //else if (it->second < threshold / 10.)
                //    break;
            }

            visualization.at<float>(y, x) = logf(freq + 1.);
            amplitude.at<float>(y, x) = logf(threshold + 1.);

            //vertical.at<float>(y, x) = logf(v + 1.);
        }

    cv::Mat vertical(visualizationRows, visualizationCols, CV_32FC1);
    for (int y = 0; y < visualizationRows; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset = y * visualizationCols + x;

            //float offsets[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];
            //float amplitudes[WINDOW_DIMENSION * WINDOW_DIMENSION - 1];

            //std::map<float, float> ordered;

            double v = 0;

            for (int j = 1; j < WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X; ++j)
            {
                if (j / WINDOW_DIMENSION_X > WINDOW_DIMENSION_Y / 2 || j % WINDOW_DIMENSION_X > WINDOW_DIMENSION_X / 2)
                    continue;

                const auto amplitude = std::abs(transformed[sourceOffset * WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X + j]);// / sqrtf(j);
                //const auto freq = hypot(j / WINDOW_DIMENSION, j % WINDOW_DIMENSION);
                //if (freq > 2)
                //    ordered[freq] = std::max(ordered[freq], amplitude);

                if (j % WINDOW_DIMENSION_X == 0)
                    v += amplitude;
            }

            vertical.at<float>(y, x) = logf(v + 1.);
        }




    cv::normalize(visualization, visualization, 0, 1, cv::NORM_MINMAX);
    cv::normalize(amplitude, amplitude, 0, 1, cv::NORM_MINMAX);
    cv::normalize(vertical, vertical, 0, 1, cv::NORM_MINMAX);

    imshow("func", func);

    imshow("visualization", visualization);

    imshow("amplitude", amplitude);

    imshow("vertical", vertical);

    cv::Mat borderline0(visualizationRows, visualizationCols, CV_8UC1, cv::Scalar(0));
    cv::Mat borderline(visualizationRows, visualizationCols, CV_8UC1, cv::Scalar(0));

    cv::Mat imgOrientationBin;
    inRange(imgOrientation, cv::Scalar(CV_PI / 2 - 0.2), cv::Scalar(CV_PI / 2 + 0.2), imgOrientationBin);


    // border line
    std::vector<cv::Point> ptSet;

    std::vector<int> lastTransitions(visualizationCols, INT_MIN / 2);

    for (int y = 0; y < visualizationRows - 1; ++y)
        for (int x = 0; x < visualizationCols; ++x)
        {
            const auto sourceOffset1 = y * visualizationCols + x;
            const auto sourceOffset2 = (y + 1) * visualizationCols + x;

            int freq1 = 0;
            int freq2 = 0;

            float threshold1 = 0;
            float threshold2 = 0;

            for (int j = 3; j <= WINDOW_DIMENSION_X / 2; ++j)
            {
                const auto amplitude1 = std::abs(transformed[sourceOffset1 * WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X + j]) / sqrt(sqrt(j));
                const auto amplitude2 = std::abs(transformed[sourceOffset2 * WINDOW_DIMENSION_Y * WINDOW_DIMENSION_X + j]) / sqrt(sqrt(j));
                if (amplitude1 > threshold1)
                {
                    freq1 = j;
                    threshold1 = amplitude1;
                }
                if (amplitude2 > threshold2)
                {
                    freq2 = j;
                    threshold2 = amplitude2;
                }
            }
            //if (freq1 > 2 && freq1 >= ((freq2 * 3 / 5 - 1)) && freq1 <= ((freq2 * 3 / 5 + 1)))
            if (freq2 > freq1 && freq2 >= freq1 * 5 / 3 && freq2 <= freq1 * 3)//5 / 2)
            {
                const auto coherency = imgCoherency.at<float>(y + WINDOW_DIMENSION_Y / 2, x + WINDOW_DIMENSION_X / 2);

                const auto orientationOk = imgOrientationBin.at<uchar>(y + WINDOW_DIMENSION_Y / 2, x + WINDOW_DIMENSION_X / 2);

                if (coherency > 0.2 && orientationOk && y - lastTransitions[x] > 100) {
                    lastTransitions[x] = y;
                    //borderline0.at<uchar>(y, x) = 255;
                    ptSet.push_back({ x, y });
                }
            }
        }

    //double A, B, C;

    //cv::Mat filtimg;
    //cv::boxFilter(borderline0, filtimg, -1, cv::Size(100, 100));// , cv::Point(0, 0), false);
    //cv::Point min_loc, max_loc;
    //double min, max;
    //cv::minMaxLoc(filtimg, &min, &max, &min_loc, &max_loc);
    //cv::rectangle(borderline0, cv::Rect(max_loc.x, max_loc.y, 100, 100), cv::Scalar(255));


    //for (auto& v : ptSet)
    //{
    //    v.y -= 1;
    //}

    // filtering
#if 1
    for (;;)
    {
        PointsProvider provider(ptSet);

        my_kd_tree_t infos(2, provider);

        infos.buildIndex();

        const int k = 4;

        std::vector<size_t> index(k);
        std::vector<float> dist(k);

        std::vector <bool> goodOnes(ptSet.size());

        for (int i = 0; i < ptSet.size(); ++i)
        {

            float pos[2];

            pos[0] = ptSet[i].x;
            pos[1] = ptSet[i].y;

            infos.knnSearch(&pos[0], k, &index[0], &dist[0]);

            goodOnes[i] = dist[k-1] < 20 * 20;
        }

        bool found = false;
        for (int i = ptSet.size(); --i >= 0;)
        {
            if (!goodOnes[i])
            {
                found = true;
                ptSet.erase(ptSet.begin() + i);
            }
        }
        if (!found)
            break;
    }
#endif

#if 1
    {
        // partition via our partitioning function
        std::vector<int> labels;
        int equilavenceClassesCount = cv::partition(ptSet, labels,
            [](const cv::Point& p1, const cv::Point& p2) {
            return hypot(p2.x - p1.x, p2.y - p1.y) < 25;
            });

        std::vector<int> groupCounts(equilavenceClassesCount);

        for (auto& l : labels)
            ++groupCounts[l];

        //auto maxIdx = std::max_element(groupCounts.begin(), groupCounts.end()) - groupCounts.begin();
        const auto threshold = *std::max_element(groupCounts.begin(), groupCounts.end()) * 0.4;

        for (int i = ptSet.size(); --i >= 0;)
        {
            if (groupCounts[labels[i]] < threshold)
                ptSet.erase(ptSet.begin() + i);
        }
    }
#endif

    for (auto& v : ptSet)
        borderline0.at<uchar>(v.y, v.x) = 255;


    cv::Mat poly;

    enum { n_samples = 8 };

    double bestCost = 1.e38;

    for (int n_ransac_samples = 1; n_ransac_samples <= n_samples; ++n_ransac_samples)
    {

        //*
        cv::Mat A;
        std::vector<bool> inliers;
        fitLineRANSAC2(ptSet, A, n_ransac_samples, //A, B, C,
            inliers);
        //*/


        for (int i = 0; i < n_samples - n_ransac_samples; ++i)
            A.push_back(0.);

        //*
        //cv::Mat A(n_samples, 1, CV_64FC1, 0.);
        std::vector<double*> params;
        for (int i = 0; i < n_samples; ++i)
            params.push_back(&A.at<double>(i, 0));

        ceres::Problem problem;
        for (int i = 0; i < ptSet.size(); ++i)
        {
            auto cost_function
                = new ceres::DynamicAutoDiffCostFunction<PolynomialResidual>(
                    new PolynomialResidual(ptSet[i].x * POLY_COEFF, ptSet[i].y, n_samples));

            //cost_function->AddParameterBlock(params.size());
            for (int j = 0; j < params.size(); ++j)
                cost_function->AddParameterBlock(1);

            cost_function->SetNumResiduals(1);

            problem.AddResidualBlock(cost_function,
                new ceres::ArctanLoss(5.),
                params);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        options.max_num_iterations = 1000;

        //options.max_linear_solver_iterations = 1000;
        //options.min_linear_solver_iterations = 950;

        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);

        if (summary.final_cost < bestCost)
        {
            bestCost = summary.final_cost;
            poly = A;
        }

        //std::cout << summary.BriefReport() << "\n";
        //*/

        //for (int i = 0; i < n_samples; ++i)
        //    std::cout << A.at<double>(i, 0) << '\n';

    }
    /*
    //cv::Mat A(n_samples, 1, CV_64FC1, 0.);

    ceres::Problem problem;
    for (int i = 0; i < ptSet.size(); ++i)
    {
        auto cost_function
            = new ceres::AutoDiffCostFunction<CrappyResidual, 8, 1, 1, 1, 1, 1, 1, 1, 1>(
                new CrappyResidual(ptSet[i].x * POLY_COEFF, ptSet[i].y));

        problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(5.),
            &A.at<double>(0, 0), &A.at<double>(1, 0), &A.at<double>(2, 0), &A.at<double>(3, 0), &A.at<double>(4, 0), &A.at<double>(5, 0), &A.at<double>(6, 0), &A.at<double>(7, 0));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    //*/


#if 0
    {
        std::vector<cv::Point2d> ptSet2;
        for (unsigned int i = 0; i < ptSet.size(); ++i) {
            if (inliers[i])
                ptSet2.push_back(ptSet[i]);
        }

        std::vector<bool> inliers2;
        fitLineRANSAC2(ptSet2, A, n_samples, //A, B, C,
            inliers2);

        std::swap(ptSet, ptSet2);
        std::swap(inliers, inliers2);
    }
#endif

    /*
    cv::Mat borderline(visualizationRows, visualizationCols, CV_8UC1, cv::Scalar(0));
    for (unsigned int i = 0; i < ptSet.size(); ++i) {
        if (inliers[i])
            borderline.at<uchar>(ptSet[i].y, ptSet[i].x) = 255;
    }
    imshow("borderline", borderline);
    */


    int x_min = INT_MAX, x_max = INT_MIN;
    // limits
    for (auto& v : ptSet)
    {
        auto y = CalcPoly(poly, v.x * POLY_COEFF);
        if (fabs(y - v.y) < 15)
        {
            x_min = std::min(x_min, v.x);
            x_max = std::max(x_max, v.x);
        }
    }



    auto surf = cv::xfeatures2d::SURF::create(1700);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    surf->detectAndCompute(func, cv::noArray(), keypoints, descriptors);

    // http://itnotesblog.ru/note.php?id=271
    cv::FlannBasedMatcher matcher;
    std::vector< cv::DMatch > matches;
    matcher.match(descriptors, GetKnownGood(), matches);

    std::vector< cv::KeyPoint > goodkeypoints;

    for (int i = 0; i < descriptors.rows; i++) {
        if (matches[i].distance < 0.33)
        {
            double y = CalcPoly(poly, std::clamp(keypoints[i].pt.x - WINDOW_DIMENSION_X / 2, float(x_min), float(x_max)) * POLY_COEFF) + WINDOW_DIMENSION_Y / 2;
            if (fabs(y - keypoints[i].pt.y) < 50)
                goodkeypoints.push_back(keypoints[i]);
        }
    }

    for (int i = goodkeypoints.size() - 1; --i >= 0;)
        for (int j = goodkeypoints.size(); --j > i;)
        {
            if (hypot(goodkeypoints[i].pt.x - goodkeypoints[j].pt.x, goodkeypoints[i].pt.y - goodkeypoints[j].pt.y) < 5)
            {
                goodkeypoints.erase(goodkeypoints.begin() + j);
            }
        }


#if 0
    std::vector<int> labels;
    int equilavenceClassesCount = cv::partition(goodkeypoints, labels, [](const cv::KeyPoint& k1, const cv::KeyPoint& k2) {
        const auto MAX_DIST = 15;
        if (/*fabs(k1.pt.x - k2.pt.x) > MAX_DIST ||*/ fabs(k1.pt.y - k2.pt.y) > MAX_DIST)
            return false;

        auto[minSize, maxSize] = std::minmax(k1.size, k2.size);
        if (maxSize / minSize > 1.35)
            return false;

        return true;
    });

    std::vector < std::vector<cv::KeyPoint>> outKeypoints(equilavenceClassesCount);
    for (int i = 0; i < goodkeypoints.size(); ++i)
    {
        outKeypoints[labels[i]].push_back(goodkeypoints[i]);
    }

    cv::RNG rng(215526);
    for (auto& kp : outKeypoints)
    {
        auto color = cv::Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));
        cv::drawKeypoints(func, kp, func, color);// , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
#endif


    auto color = cv::Scalar(0, 255, 0);
    cv::drawKeypoints(func, goodkeypoints, func, color);// , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


    imshow("borderline0", borderline0);


    std::vector<cv::Point> points_fitted;
    for (int x = 0; x < visualizationCols; x++)
    {
        //double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
        //    A.at<double>(2, 0)*std::pow(x, 2) + A.at<double>(3, 0)*std::pow(x, 3);

        //double y = poly.at<double>(0, 0) + poly.at<double>(1, 0) * x * POLY_COEFF;
        //for (int i = 2; i < n_samples; ++i)
        //    y += poly.at<double>(i, 0) * std::pow(x * POLY_COEFF, i);

        double y = CalcPoly(poly, std::clamp(x, x_min, x_max) * POLY_COEFF);

        points_fitted.push_back(cv::Point(x + WINDOW_DIMENSION_X / 2, y + WINDOW_DIMENSION_Y / 2));
    }

    cv::polylines(func, points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
    imshow("image", func);

    cv::Mat theMask(IMAGE_DIMENSION, IMAGE_DIMENSION, CV_8UC1, cv::Scalar(0));

    std::vector<std::vector<cv::Point> > fillContAll;
    fillContAll.push_back(points_fitted);

    fillContAll[0].push_back(cv::Point(IMAGE_DIMENSION - WINDOW_DIMENSION_X / 2, 0));
    fillContAll[0].push_back(cv::Point(WINDOW_DIMENSION_X / 2, 0));

    cv::fillPoly(theMask, fillContAll, cv::Scalar(255));

    imshow("theMask", theMask);

    cv::Mat imgCoherencyBin = imgCoherency > 0.2;

    imshow("imgCoherencyBin", imgCoherencyBin);
    imshow("imgOrientationBin", imgOrientationBin);


    //imgCoherency *= 10;
    //cv::exp(imgCoherency, imgCoherency);

    cv::normalize(imgCoherency, imgCoherency, 0, 1, cv::NORM_MINMAX);
    cv::normalize(imgOrientation, imgOrientation, 0, 1, cv::NORM_MINMAX);

    imshow("imgCoherency", imgCoherency);
    imshow("imgOrientation", imgOrientation);





///////////////////////////////////////////////////////////////////////////////

    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 2.);

    dst &= imgCoherencyBin;

    dst &= mask;

    dst &= diff;

    imshow("Dst before", dst);

    cv::ximgproc::thinning(dst, dst);

    auto skeleton = dst.clone();

    //dst &= theMask;

    imshow("Thinning", dst);

    // Specify size on vertical axis
    int vertical_size = 5;// dst.rows / 30;
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    // Apply morphology operations
    erode(dst, dst, verticalStructure);
    dilate(dst, dst, verticalStructure);


    //Canny(dst, dst, 50, 200, 3);

    //cv::Rect roi(50, 50, dst.cols - 100, dst.rows - 100);
    //dst = dst(roi);


#if 0
    // Declare the variables we are going to use
    Mat src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    GaussianBlur(src, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
    Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    // converting back to CV_8U
    convertScaleAbs(dst, dst);
#endif

    //![edge_detection]

    // Copy edges to the images that will display the results in BGR
    Mat cdstP;
    cvtColor(dst, cdstP, COLOR_GRAY2BGR);
    //cdstP = cdst.clone();

    /*
    //![hough_lines]
    // Standard Hough Line Transform
    Mat cdst = cdstP.clone();
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI / 180, 200, 0, 0, 0, 0.1);// CV_PI / 2 - 0.2, CV_PI / 2 + 0.2 ); // runs the actual detection
    //HoughLines(dst, lines, 1, CV_PI / 180, 800); // runs the actual detection
    //![hough_lines]
    //![draw_lines]
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( cdst, pt1, pt2, Scalar(0,0,255), 1, LINE_AA);
    }
    //![draw_lines]
    //*/

    //![hough_lines_p]
    // Probabilistic Line Transform
    std::vector<cv::Vec4i> linesP; // will hold the results of the detection
    const int threshold = 10;
    HoughLinesP(dst, linesP, 1, CV_PI / 180 / 10, threshold, 5, 25); // runs the actual detection
    //![hough_lines_p]
    //![draw_lines_p]
    // Draw the lines
    //for( size_t i = 0; i < linesP.size(); i++ )

    linesP.erase(std::remove_if(linesP.begin(), linesP.end(), [&dst](const Vec4i& l) {
        const double expectedAlgle = 0;
        const double expectedAngleDiff = 1.;
        const auto border = 10;
        return l[1] == l[3] || fabs(double(l[0] - l[2]) / (l[1] - l[3]) + expectedAlgle) > expectedAngleDiff
            || l[0] < border && l[2] < border || l[1] == 0 && l[3] == 0
            || l[0] >= (dst.cols - border) && l[2] >= (dst.cols - border) || l[1] == dst.rows - 1 && l[3] == dst.rows - 1;
    }), linesP.end());


    //auto kingThreshold = std::min(
    //    CalcPoly(poly, x_min * POLY_COEFF) + WINDOW_DIMENSION_Y / 2,
    //    CalcPoly(poly, x_max * POLY_COEFF) + WINDOW_DIMENSION_Y / 2) - 120;

    //linesP.erase(std::remove_if(linesP.begin(), linesP.end(), [kingThreshold](const Vec4i& l) {
    //    return l[1] < kingThreshold && l[3] < kingThreshold;
    //}), linesP.end());


    auto angleSortLam = [](const Vec4i& l) {
        return double(l[0] - l[2]) / (l[1] - l[3]);
    };

    std::sort(linesP.begin(), linesP.end(), [&angleSortLam](const Vec4i& l1, const Vec4i& l2) {
        return angleSortLam(l1) < angleSortLam(l2);
    });

    const double maxDiff = 0.1;

    auto itFirst = linesP.begin();
    auto itLast = linesP.begin();

    double sum = 0;
    double maxSum = 0;

    auto itBegin = linesP.begin();
    auto itEnd = linesP.begin();

    while (itFirst != linesP.end() && itLast != linesP.end())
    {
        auto start = angleSortLam(*itFirst);

        while (itLast != linesP.end() && angleSortLam(*itLast) < start + maxDiff)
        {
            sum += hypot((*itLast)[0] - (*itLast)[2], (*itLast)[1] - (*itLast)[3]);
            ++itLast;
        }
        if (sum > maxSum)
        {
            itBegin = itFirst;
            itEnd = itLast;
            maxSum = sum;
        }


        sum -= hypot((*itFirst)[0] - (*itFirst)[2], (*itFirst)[1] - (*itFirst)[3]);
        ++itFirst;
    }

    // vector<Vec4i>
    linesP = { itBegin, itEnd };


    //for (int i = linesP.size(); --i >= 0; )
    //{
    //    Vec4i l = linesP[i];
    //    const double expectedAlgle = 0.05;
    //    const auto border = 10;
    //    if (l[1] == l[3] || fabs(double(l[0] - l[2]) / (l[1] - l[3]) + expectedAlgle) > expectedAlgle
    //        || l[0] < border && l[2] < border || l[1] == 0 && l[3] == 0
    //        || l[0] >= (dst.cols - border) && l[2] >= (dst.cols - border) || l[1] == dst.rows - 1 && l[3] == dst.rows - 1)
    //    //if (l[1] == l[3] || fabs(double(l[0] - l[2]) / (l[1] - l[3])) > 0.1)
    //    {
    //        linesP.erase(linesP.begin() + i);
    //        continue;
    //    }

    //    auto color = (min(l[1], l[3]) < 380) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

    //    line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), color, 1, LINE_AA);
    //}
    //![draw_lines_p]

    for (Vec4i& l : linesP) {
        auto color = (min(l[1], l[3]) < 380) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
        line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), color, 1, LINE_AA);
    }

    /////////////////////////////////////////////////////////////////////////////////////
#if 0
    //// remove small lines
    //std::vector<Vec4i> linesWithoutSmall;
    //std::copy_if(linesP.begin(), linesP.end(), std::back_inserter(linesWithoutSmall), [](Vec4f line) {
    //    float length = sqrtf((line[2] - line[0]) * (line[2] - line[0])
    //        + (line[3] - line[1]) * (line[3] - line[1]));
    //    return length > 5;
    //});

    //std::cout << "Detected: " << linesWithoutSmall.size() << std::endl;

    // partition via our partitioning function
    std::vector<int> labels;
    int equilavenceClassesCount = cv::partition(linesP, labels, [](const Vec4i& l1, const Vec4i& l2) {
        return extendedBoundingRectangleLineEquivalence(
            l1, l2,
            // line extension length
            25,
            // line extension length - as fraction of original line width
            1.0,
            // thickness of bounding rectangle around each line
            3);
    });

    //std::cout << "Equivalence classes: " << equilavenceClassesCount << std::endl;

    Mat detectedLinesImg = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
    Mat reducedLinesImg = detectedLinesImg.clone();

    std::vector<std::vector<Vec4i>> groups(equilavenceClassesCount);
    for (int i = 0; i < linesP.size(); i++) {
        Vec4i& detectedLine = linesP[i];
        groups[labels[i]].push_back(detectedLine);
    }

#if 0
    for (int i = groups.size(); --i >= 0; ) {
        auto& group = groups[i];
        int minX = INT_MAX;
        int minY = INT_MAX;
        int maxX = INT_MIN;
        int maxY = INT_MIN;
        int length = 0;

        for (auto& line : group) {
            minX = std::min({ minX, line[0], line[2] });
            maxX = std::max({ maxX, line[0], line[2] });
            minY = std::min({ minY, line[1], line[3] });
            maxY = std::max({ maxY, line[1], line[3] });

            length += hypot(line[2] - line[0], line[3] - line[1]);
        }

        const auto extent = hypot(maxX - minX, maxY - minY);

        if (length > extent * 1.5) { // split

            std::vector<Point2i> pointCloud;
            for (auto& detectedLine : group) {
                pointCloud.push_back(Point2i(detectedLine[0], detectedLine[1]));
                pointCloud.push_back(Point2i(detectedLine[2], detectedLine[3]));
            }

            Vec4f lineParams;
            fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);

            const auto cos_phi = -lineParams[1];
            const auto sin_phi = lineParams[0];

            std::vector<double> offsets;
            for (auto& detectedLine : group) {
                double x = (detectedLine[0] + detectedLine[2]) / 2.;
                double y = (detectedLine[1] + detectedLine[3]) / 2.;
                double x_new = x * cos_phi - y * sin_phi;
                offsets.push_back(x_new);
            }

            const auto minmax = std::minmax_element(offsets.begin(), offsets.end());
            const auto medium = (*minmax.first + *minmax.second) / 2;

            std::vector<Vec4i> first, second;
            for (int i = 0; i < group.size(); ++i) {
                auto& line = group[i];
                if (offsets[i] < medium)
                    first.push_back(line);
                else
                    second.push_back(line);
            }

            group = first;
            groups.push_back(second);
        }
    }
#endif

    equilavenceClassesCount = groups.size();

    // grab a random colour for each equivalence class
    RNG rng(215526);
    std::vector<Scalar> colors(equilavenceClassesCount);
    for (int i = 0; i < equilavenceClassesCount; i++) {
        colors[i] = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
    }

    // draw original detected lines
    //for (int i = 0; i < linesP.size(); i++) {
    for (int i = 0; i < equilavenceClassesCount; ++i)
        for (auto &detectedLine : groups[i]) {
            //Vec4i& detectedLine = linesP[i];
            line(detectedLinesImg,
                cv::Point(detectedLine[0], detectedLine[1]),
                cv::Point(detectedLine[2], detectedLine[3]), colors[i/*labels[i]*/], 2);
        }

    // build point clouds out of each equivalence classes
    std::vector<std::vector<Point2i>> pointClouds(equilavenceClassesCount);
    //for (int i = 0; i < linesP.size(); i++) {
    for (int i = 0; i < equilavenceClassesCount; ++i)
        for (auto &detectedLine : groups[i]) {
            //Vec4i& detectedLine = linesP[i];
            pointClouds[i/*labels[i]*/].push_back(Point2i(detectedLine[0], detectedLine[1]));
            pointClouds[i/*labels[i]*/].push_back(Point2i(detectedLine[2], detectedLine[3]));
        }

    // fit line to each equivalence class point cloud
    //std::vector<Vec4i> reducedLines = std::accumulate(
    //    pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{}, [](std::vector<Vec4i> target, const std::vector<Point2i>& _pointCloud) {
    //    std::vector<Point2i> pointCloud = _pointCloud;

    //    //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
    //    // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
    //    Vec4f lineParams;
    //    fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);

    //    // derive the bounding xs of point cloud
    //    decltype(pointCloud)::iterator minXP, maxXP;
    //    std::tie(minXP, maxXP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const Point2i& p1, const Point2i& p2) { return p1.x < p2.x; });

    //    // derive y coords of fitted line
    //    float m = lineParams[1] / lineParams[0];
    //    int y1 = ((minXP->x - lineParams[2]) * m) + lineParams[3];
    //    int y2 = ((maxXP->x - lineParams[2]) * m) + lineParams[3];

    //    target.push_back(Vec4i(minXP->x, y1, maxXP->x, y2));
    //    return target;
    //});
    std::vector<Vec4i> reducedLines = std::accumulate(
        pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{}, [](std::vector<Vec4i> target, const std::vector<Point2i>& _pointCloud) {
        std::vector<Point2i> pointCloud = _pointCloud;

        //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
        // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
        Vec4f lineParams;
        fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);

        // derive the bounding xs of point cloud
        decltype(pointCloud)::iterator minYP, maxYP;
        std::tie(minYP, maxYP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const Point2i& p1, const Point2i& p2) { return p1.y < p2.y; });

        // derive y coords of fitted line
        float m = lineParams[0] / lineParams[1];
        int x1 = ((minYP->y - lineParams[3]) * m) + lineParams[2];
        int x2 = ((maxYP->y - lineParams[3]) * m) + lineParams[2];

        target.push_back(Vec4i(x1, minYP->y, x2, maxYP->y));
        return target;
    });

    //for (Vec4i reduced : reducedLines) {
    //    line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), Scalar(255, 255, 255), 2);
    //}

    for (int i = 0; i < reducedLines.size(); ++i)
    {
        auto& reduced = reducedLines[i];
        line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), colors[i], 2);
    }
#endif

    auto reducedLines0 = reduceLines(linesP, 25, 1.0, 3);

    {
        // find prevailing direction
        std::vector<Point2i> pointCloud;
        for (auto& reduced : reducedLines0)
        {
            auto centerX = (reduced[0] + reduced[2]) / 2;
            auto centerY = (reduced[1] + reduced[3]) / 2;
            pointCloud.emplace_back(reduced[0] - centerX, reduced[1] - centerY);
            pointCloud.emplace_back(reduced[2] - centerX, reduced[3] - centerY);
        }
        Vec4f lineParams;
        fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);
        //const auto cos_phi = -lineParams[1];
        //const auto sin_phi = -lineParams[0];
        const auto tan_phi = lineParams[0] / lineParams[1];

        reducedLines0.erase(std::remove_if(reducedLines0.begin(), reducedLines0.end(), [tan_phi](const Vec4i& line) {
            return hypot(line[2] - line[0], line[3] - line[1]) <= 10
                || fabs(double(line[2] - line[0]) / (line[3] - line[1]) - tan_phi) > 0.05
                ;
        }), reducedLines0.end());
    }

    auto reducedLines = reduceLines(reducedLines0, 50, 0.7, 4);

    //reducedLines.erase(std::remove_if(reducedLines.begin(), reducedLines.end(), [](const Vec4i& line) {
    //    return hypot(line[2] - line[0], line[3] - line[1]) <= 10;
    //}), reducedLines.end());

    Mat reducedLinesImg0 = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
    {
        RNG rng(215526);
        for (auto & reduced : reducedLines0)
        {
            auto color = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
            line(reducedLinesImg0, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), color, 2);
        }
    }

    //Mat reducedLinesImg = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
    //{
    //    RNG rng(215526);
    //    for (int i = 0; i < reducedLines.size(); ++i)
    //    {
    //        auto color = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
    //        auto& reduced = reducedLines[i];
    //        line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), color, 2);
    //    }
    //}

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

    std::sort(reducedLines.begin(), reducedLines.end(), [&sortLam](const Vec4i& l1, const Vec4i& l2) {
        return sortLam(l1) < sortLam(l2);
    });

    auto approveLam = [](const Vec4i& line) {
        return hypot(line[2] - line[0], line[3] - line[1]) > 100;
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

    // garbage in garbage out
    int y0 = INT_MAX;
    int i0 = 0;
    for (int i = 0; i < reducedLines.size(); ++i) {
        int y = reducedLines[i][1] + reducedLines[i][0];
        if (y < y0) {
            y0 = y;
            i0 = i;
        }
    }
    if (!(i0 & 1)) {
        for (int i = 0; i < i0; i += 2)
        {
            auto& line = reducedLines[i];
            const int y = y0 - line[0];
            if (line[1] > y) {
                line[0] = line[2] + double(line[0] - line[2]) / (line[1] - line[3]) * (y - line[3]);
                line[1] = y;
            }
        }
    }

    // Cutting lines
    const auto removeThreshold = 40;
    for (int i = reducedLines.size(); --i >= 0;)
    {
        auto& line = reducedLines[i];
        double x = (line[0] + line[2]) / 2.;
        double y = CalcPoly(poly, std::clamp(x - WINDOW_DIMENSION_X / 2, double(x_min), double(x_max)) * POLY_COEFF) + WINDOW_DIMENSION_Y / 2;
        if (y > line[3])
            continue;
        if (y < line[1] + removeThreshold)
        {
            reducedLines.erase(reducedLines.begin() + i);
        }
        else
        {
            line[2] = line[0] + double(line[2] - line[0]) / (line[3] - line[1]) * (y - line[1]);
            line[3] = y;
        }
    }


    Mat reducedLinesImg = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
    {
        RNG rng(215526);
        for (auto & reduced : reducedLines)
        {
            auto color = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
            line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), color, 2);
        }
    }

    for (int i = 0; i < int(reducedLines.size()) - 1; ++i) {
        auto& first = reducedLines[i];
        auto& second = reducedLines[i + 1];
        if (first[1] > second[1]) {
            continue;
        }

        const auto y_first = first[0] * sin_phi + first[1] * cos_phi;
        const auto y_second = second[0] * sin_phi + second[1] * cos_phi;

        const auto x_first = first[0] * cos_phi - first[1] * sin_phi;
        const auto x_second = second[0] * cos_phi - second[1] * sin_phi;

        const auto diff = y_second - y_first;

        std::cout << (diff * 1.27) << '\n';
    }


    //////////////////////////////////////////////////////////////////////////

    /*

    // sort
    auto keyPointsSortLam = [cos_phi, sin_phi](const KeyPoint& kp) {
        double x_new = kp.pt.x * cos_phi - kp.pt.y * sin_phi;
        return x_new;
    };

    std::sort(goodkeypoints.begin(), goodkeypoints.end(), [&keyPointsSortLam](const KeyPoint& kp1, const KeyPoint& kp2) {
        return keyPointsSortLam(kp1) < keyPointsSortLam(kp2);
    });

    */

    // turtle stuff

    std::deque<std::pair<Point, Point>> turtleLines;

    const double correction_coeff = 0.2;

    //for (auto& kp : goodkeypoints)
    for (int i = goodkeypoints.size(); --i >= 0;)
    {
        auto& kp = goodkeypoints[i];
        cv::Point pos(kp.pt);
        auto start = FindPath(skeleton, pos);

        if (start.y > pos.y - 10)
        {
            goodkeypoints.erase(goodkeypoints.begin() + i);
            continue;
        }

        pos.x += kp.size * sin_phi * correction_coeff;
        pos.y += kp.size * cos_phi * correction_coeff;

        turtleLines.emplace_front(start, pos);
    }

    for (int i = goodkeypoints.size(); --i >= 0;)
    {
        bool erase = false;
        for (int j = goodkeypoints.size(); --j >= 0;)
        {
            if (i == j)
                continue;
            if (turtleLines[i].first == turtleLines[j].first && abs(turtleLines[i].second.x - turtleLines[j].second.x) < 10)
            {
                double y_i = CalcPoly(poly, std::clamp(turtleLines[i].second.x - WINDOW_DIMENSION_X / 2, x_min, x_max) * POLY_COEFF) + WINDOW_DIMENSION_Y / 2;
                double y_j = CalcPoly(poly, std::clamp(turtleLines[j].second.x - WINDOW_DIMENSION_X / 2, x_min, x_max) * POLY_COEFF) + WINDOW_DIMENSION_Y / 2;
                if (abs(y_j - turtleLines[j].second.y) < abs(y_i - turtleLines[i].second.y))
                {
                    erase = true;
                    break;
                }
            }
        }
        if (erase)
        {
            goodkeypoints.erase(goodkeypoints.begin() + i);
            turtleLines.erase(turtleLines.begin() + i);
        }
    }

    cv::Mat outSkeleton;
    cv::drawKeypoints(skeleton, goodkeypoints, outSkeleton, { 0, 255, 0 });// , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    for (auto& l : turtleLines)
    {
        int radius = 2;
        int thickness = -1;
        circle(outSkeleton, l.first, radius, { 0, 255, 0 }, thickness);
        line(outSkeleton, l.second, l.first, { 0, 255, 0 });
    }

    //for (auto& kp : goodkeypoints)
    //{
    //    cv::Point pos(kp.pt);
    //    auto start = FindPath(skeleton, pos);
    //    int radius = 2;
    //    int thickness = -1;
    //    circle(outSkeleton, start, radius, { 0, 255, 0 }, thickness);
    //    line(outSkeleton, pos, start, { 0, 255, 0 });
    //    /*
    //    const auto y_first = start.x * sin_phi + start.y * cos_phi;
    //    const auto y_second = pos.x * sin_phi + pos.y * cos_phi;

    //    const auto x_first = start.x * cos_phi - start.y * sin_phi;
    //    const auto x_second = pos.x * cos_phi - pos.y * sin_phi;

    //    line(outSkeleton, Point(x_first, y_first), Point(x_second, y_second), { 255, 0, 0 });
    //    */
    //}

    imshow("outSkeleton", outSkeleton);


    // Merge turtleLines and reducedLines

    std::vector<std::tuple<double, double, double, double, double>> result;

    for (int i = 0; i < reducedLines.size() - 1; ++i) {
        auto& first = reducedLines[i];
        auto& second = reducedLines[i + 1];
        //if (first[1] > second[1]) {
        //    continue;
        //}

        auto y_first = first[1];
        auto x_first = first[0];

        auto y_second = (first[3] + second[3]) / 2.;
        auto x_second = (first[2] + second[2]) / 2.;


        for (auto& l : turtleLines)
        {
            if (l.second.x > first[2] && l.second.x < second[2])
            {
                y_first = l.first.y;
                x_first = l.first.x;
                y_second = l.second.y;
                x_second = l.second.x;

                break;
            }
        }


        const auto y_first_rotated = first[0] * sin_phi + first[1] * cos_phi;
        const auto y_second_rotated = x_second * sin_phi + y_second * cos_phi;

        //const auto x_first = first[0] * cos_phi - first[1] * sin_phi;
        //const auto x_second = second[0] * cos_phi - second[1] * sin_phi;

        const auto diff = (y_second_rotated - y_first_rotated) * originalSize.height / IMAGE_DIMENSION;

        auto convertX = [&originalSize, mirrorX](int x) {
            int result = x * originalSize.width / IMAGE_DIMENSION;
            if (mirrorX)
                result = originalSize.width - 1 - result;
            return result;
        };
        auto convertY = [&originalSize, mirrorY](int y) {
            int result = y * originalSize.height / IMAGE_DIMENSION;
            if (mirrorY)
                result = originalSize.height - 1 - result;
            return result;
        };

        result.push_back({ convertX(x_first), convertY(y_first), convertX(x_second), convertY(y_second), diff });
    }

    //auto qrwer = skeleton.clone();
    Mat qrwer = Mat::zeros(originalSize.height, originalSize.width, CV_8UC3);

    for (auto& line : result) {
        Scalar color = Scalar(0, 255, 0);
        //Scalar color = Scalar(255);
        int radius = 2;
        int thickness = -1;
        circle(qrwer, Point(std::get<0>(line), std::get<1>(line)), radius, color, thickness);
        circle(qrwer, Point(std::get<2>(line), std::get<3>(line)), radius, color, thickness);
        cv::line(qrwer, Point(std::get<0>(line), std::get<1>(line)), Point(std::get<2>(line), std::get<3>(line)), color);
    }

    imshow("qrwer", qrwer);


    //imshow("funkyFunc", funkyFunc);

    //imshow("level", level);


    //imshow("lowLevel", lowLevel);
    ////imshow("funkyLowLevel", funkyLowLevel);



    //imshow("Filtered", filtered);

    imshow("Mask", mask);
    imshow("Diff", diff);

    imshow("Transform", dst);

    //imshow("Threshold0", thresh0);

    //imshow("Threshold", thresh);

    //imshow("Detected Lines (in red) - Line Transform", cdst);

    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    //![imshow]


    //imshow("Detected Lines", detectedLinesImg);
    imshow("Reduced Lines 0", reducedLinesImg0);
    imshow("Reduced Lines", reducedLinesImg);


    return result;
}
