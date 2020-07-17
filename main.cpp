/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/ximgproc.hpp"

#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;


using namespace cv::ximgproc;


// Rearrange the quadrants of a Fourier image so that the origin is at
// the image center

void shiftDFT(Mat& fImage)
{
    Mat tmp, q0, q1, q2, q3;

    // first crop the image, if it has an odd number of rows or columns

    fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

    int cx = fImage.cols / 2;
    int cy = fImage.rows / 2;

    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center

    q0 = fImage(Rect(0, 0, cx, cy));
    q1 = fImage(Rect(cx, 0, cx, cy));
    q2 = fImage(Rect(0, cy, cx, cy));
    q3 = fImage(Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/******************************************************************************/
// return a floating point spectrum magnitude image scaled for user viewing
// complexImg- input dft (2 channel floating point, Real + Imaginary fourier image)
// rearrange - perform rearrangement of DFT quadrants if true

// return value - pointer to output spectrum magnitude image scaled for user viewing

Mat create_spectrum_magnitude_display(Mat& complexImg, bool rearrange)
{
    Mat planes[2];

    // compute magnitude spectrum (N.B. for display)
    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))

    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);

    Mat mag = (planes[0]).clone();
    mag += Scalar::all(1);
    log(mag, mag);

    if (rearrange)
    {
        // re-arrange the quaderants
        shiftDFT(mag);
    }

    normalize(mag, mag, 0, 1, NORM_MINMAX);

    return mag;

}
/******************************************************************************/

// create a 2-channel butterworth low-pass filter with radius D, order n
// (assumes pre-aollocated size of dft_Filter specifies dimensions)

void create_butterworth_lowpass_filter(Mat &dft_Filter, int D, int n)
{
    Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

    Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
    double radius;

#if 0
    // based on the forumla in the IP notes (p. 130 of 2009/10 version)
    // see also HIPR2 on-line

    for (int i = 0; i < dft_Filter.rows; i++)
    {
        for (int j = 0; j < dft_Filter.cols; j++)
        {
            radius = (double)sqrt(pow((i - centre.x), 2.0) + pow((double)(j - centre.y), 2.0));
            tmp.at<float>(i, j) = (float)
                (1 / (1 + pow((double)(radius / D), (double)(2 * n))));
        }
    }
#endif

    for (int i = 0; i < dft_Filter.rows; i++)
    {
        for (int j = 0; j < dft_Filter.cols; j++)
        {
            float v = 0;
            const double radius2d = (double)sqrt(pow((i - centre.x), 2.0) + pow((double)(j - centre.y), 2.0));
            if (radius2d > 0)
            {
                radius = fabs(i - centre.x);
                const auto d = D / 10;
                v = (1 / (1 + pow((double)(radius / D), (double)(2 * n))));
                v *= (1 / (1 + pow((double)(d / radius2d), (double)(2 * n))));
            }
            tmp.at<float>(i, j) = v;
        }
    }


    Mat toMerge[] = { tmp, tmp };
    merge(toMerge, 2, dft_Filter);
}


void filter2DFreq(const Mat& inputImg, Mat& outputImg/*, const Mat& H*/)
{
    //int radius = 50;				// low pass filter parameter
    int radius = 10;				// low pass filter parameter
    int order = 2;				// low pass filter parameter

    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);

    auto filter = complexI.clone();
    create_butterworth_lowpass_filter(filter, radius, order);

    //Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    //Mat complexH;
    //merge(planesH, 2, complexH);
    //Mat complexIH;
    //mulSpectrums(complexI, complexH, complexIH, 0);

    // apply filter
    shiftDFT(complexI);
    mulSpectrums(complexI, filter, complexI, 0);
    shiftDFT(complexI);


    idft(complexI, complexI);
    split(complexI, planes);
    outputImg = planes[0];
}

/////////////////////////////////////////////////////////////////////////////////////////

// https://stackoverflow.com/a/51121483

//Vec2d linearParameters(Vec4i line) {
//    Mat a = (Mat_<double>(2, 2) <<
//        line[0], 1,
//        line[2], 1);
//    Mat y = (Mat_<double>(2, 1) <<
//        line[1],
//        line[3]);
//    Vec2d mc; solve(a, y, mc);
//    return mc;
//}

Vec4i extendedLine(const Vec4i& line, double d, double max_coeff) {
    //// oriented left-t-right
    //Vec4d _line = line[2] - line[0] < 0 ? Vec4d(line[2], line[3], line[0], line[1]) : Vec4d(line[0], line[1], line[2], line[3]);
    //double m = linearParameters(_line)[0];
    //// solution of pythagorean theorem and m = yd/xd
    //double xd = sqrt(d * d / (m * m + 1));
    //double yd = xd * m;
    //return Vec4d(_line[0] - xd, _line[1] - yd, _line[2] + xd, _line[3] + yd);

    const auto length = hypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = std::min(d / length, max_coeff);
    double xd = (line[2] - line[0]) * coeff;
    double yd = (line[3] - line[1]) * coeff;
    return Vec4d(line[0] - xd, line[1] - yd, line[2] + xd, line[3] + yd);
}

std::vector<Point2i> boundingRectangleContour(const Vec4i& line, float d) {
    //// finds coordinates of perpendicular lines with length d in both line points
    //// https://math.stackexchange.com/a/2043065/183923

    //Vec2f mc = linearParameters(line);
    //float m = mc[0];
    //float factor = sqrtf(
    //    (d * d) / (1 + (1 / (m * m)))
    //);

    //float x3, y3, x4, y4, x5, y5, x6, y6;
    //// special case(vertical perpendicular line) when -1/m -> -infinity
    //if (m == 0) {
    //    x3 = line[0]; y3 = line[1] + d;
    //    x4 = line[0]; y4 = line[1] - d;
    //    x5 = line[2]; y5 = line[3] + d;
    //    x6 = line[2]; y6 = line[3] - d;
    //}
    //else {
    //    // slope of perpendicular lines
    //    float m_per = -1 / m;

    //    // y1 = m_per * x1 + c_per
    //    float c_per1 = line[1] - m_per * line[0];
    //    float c_per2 = line[3] - m_per * line[2];

    //    // coordinates of perpendicular lines
    //    x3 = line[0] + factor; y3 = m_per * x3 + c_per1;
    //    x4 = line[0] - factor; y4 = m_per * x4 + c_per1;
    //    x5 = line[2] + factor; y5 = m_per * x5 + c_per2;
    //    x6 = line[2] - factor; y6 = m_per * x6 + c_per2;
    //}

    //return std::vector<Point2i> {
    //    Point2i(x3, y3),
    //        Point2i(x4, y4),
    //        Point2i(x6, y6),
    //        Point2i(x5, y5)
    //};

    const auto length = hypot(line[2] - line[0], line[3] - line[1]);
    const auto coeff = d / length;

    // dx <= -dy
    // dy <= dx
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

    //Vec4i l1(_l1), l2(_l2);
    // extend lines by percentage of line width
    //float len1 = sqrtf((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]));
    //float len2 = sqrtf((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]));

    Vec4i el1 = extendedLine(l1, extensionLength, extensionLengthMaxFraction);
    Vec4i el2 = extendedLine(l2, extensionLength, extensionLengthMaxFraction);

    // reject the lines that have wide difference in angles
    //float a1 = atan(linearParameters(el1)[0]);
    //float a2 = atan(linearParameters(el2)[0]);
    //if (fabs(a1 - a2) > maxAngleDiff * CV_PI / 180.0) {
    //    return false;
    //}

    // calculate window around extended line
    // at least one point needs to inside extended bounding rectangle of other line,
    std::vector<Point2i> lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
    return
        pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) >= 0 ||
        pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) >= 0 ||

        pointPolygonTest(lineBoundingContour, cv::Point(l2[0], l2[1]), false) >= 0 ||
        pointPolygonTest(lineBoundingContour, cv::Point(l2[2], l2[3]), false) >= 0;

    //std::vector<Point2i> lineBoundingContour = boundingRectangleContour(el1, boundingRectangleThickness / 2);
    //if (pointPolygonTest(lineBoundingContour, cv::Point(el2[0], el2[1]), false) >= 0 ||
    //    pointPolygonTest(lineBoundingContour, cv::Point(el2[2], el2[3]), false) >= 0) {
    //    return true;
    //}

    //lineBoundingContour = boundingRectangleContour(el2, boundingRectangleThickness / 2);
    //if (pointPolygonTest(lineBoundingContour, cv::Point(el1[0], el1[1]), false) >= 0 ||
    //    pointPolygonTest(lineBoundingContour, cv::Point(el1[2], el1[3]), false) >= 0) {
    //    return true;
    //}

    //return false;
}

Vec4i HandlePointCloud(const std::vector<Point2i>& pointCloud) {
    //lineParams: [vx,vy, x0,y0]: (normalized vector, point on our contour)
    // (x,y) = (x0,y0) + t*(vx,vy), t -> (-inf; inf)
    Vec4f lineParams;
    fitLine(pointCloud, lineParams, DIST_L2, 0, 0.01, 0.01);

    // derive the bounding xs of point cloud
    std::vector<Point2i>::const_iterator minYP, maxYP;
    std::tie(minYP, maxYP) = std::minmax_element(pointCloud.begin(), pointCloud.end(), [](const Point2i& p1, const Point2i& p2) { return p1.y < p2.y; });

    // derive y coords of fitted line
    float m = lineParams[0] / lineParams[1];
    int x1 = ((minYP->y - lineParams[3]) * m) + lineParams[2];
    int x2 = ((maxYP->y - lineParams[3]) * m) + lineParams[2];

    return { x1, minYP->y, x2, maxYP->y };

}

std::vector<Vec4i> reduceLines(const vector<Vec4i>& linesP,
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

    //std::cout << "Equivalence classes: " << equilavenceClassesCount << std::endl;

    //Mat detectedLinesImg = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
    //Mat reducedLinesImg = detectedLinesImg.clone();

    std::vector<std::vector<Vec4i>> groups(equilavenceClassesCount);
    for (int i = 0; i < linesP.size(); i++) {
        const Vec4i& detectedLine = linesP[i];
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
            const auto sin_phi = -lineParams[0];

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
    //RNG rng(215526);
    //std::vector<Scalar> colors(equilavenceClassesCount);
    //for (int i = 0; i < equilavenceClassesCount; i++) {
    //    colors[i] = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
    //}

#if 0
    // draw original detected lines
    //for (int i = 0; i < linesP.size(); i++) {
    for (int i = 0; i < equilavenceClassesCount; ++i)
        for (auto &detectedLine : groups[i]) {
            //Vec4i& detectedLine = linesP[i];
            line(detectedLinesImg,
                cv::Point(detectedLine[0], detectedLine[1]),
                cv::Point(detectedLine[2], detectedLine[3]), colors[i/*labels[i]*/], 2);
        }
#endif

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
        pointClouds.begin(), pointClouds.end(), std::vector<Vec4i>{}, [](std::vector<Vec4i> target, const std::vector<Point2i>& pointCloud) {
            target.push_back(HandlePointCloud(pointCloud));
            return target;
        });

    return reducedLines;
}



int main(int argc, char** argv)
{
    // Declare the output variables
    //Mat dst, cdst, cdstP;

    //![load]
    const char* default_file = "../data/sudoku.png";
    const char* filename = argc >=2 ? argv[1] : default_file;

    // Loads an image
    Mat src = imread( filename, IMREAD_GRAYSCALE );

    // Check if image is loaded fine
    if(src.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }
    //![load]

#if 0
    Mat image;
    filter2DFreq(src, image);
    normalize(image, image, 0, 255, NORM_MINMAX);
    image.convertTo(image, CV_8U);
#endif


#if 0

    Mat image;
    filter2DFreq(src, image);
    normalize(image, image, 0, 255, NORM_MINMAX);
    image.convertTo(image, CV_8U);


    // Create FLD detector
    // Param               Default value   Description
    // length_threshold    10            - Segments shorter than this will be discarded
    // distance_threshold  1.41421356    - A point placed from a hypothesis line
    //                                     segment farther than this will be
    //                                     regarded as an outlier
    // canny_th1           50            - First threshold for
    //                                     hysteresis procedure in Canny()
    // canny_th2           50            - Second threshold for
    //                                     hysteresis procedure in Canny()
    // canny_aperture_size 3             - Aperturesize for the sobel
    //                                     operator in Canny()
    // do_merge            false         - If true, incremental merging of segments
    //                                     will be perfomred
    int length_threshold = 10;
    float distance_threshold = 1.41421356f;
    double canny_th1 = 50.0;
    double canny_th2 = 50.0;
    int canny_aperture_size = 3;

    //bool do_merge = false;
    bool do_merge = true;

    Ptr<FastLineDetector> fld = createFastLineDetector(length_threshold,
        distance_threshold, canny_th1, canny_th2, canny_aperture_size,
        do_merge);
    vector<Vec4f> lines_fld;
    // Because of some CPU's power strategy, it seems that the first running of
    // an algorithm takes much longer. So here we run the algorithm 10 times
    // to see the algorithm's processing time with sufficiently warmed-up
    // CPU performance.
    for (int run_count = 0; run_count < 10; run_count++) {
        double freq = getTickFrequency();
        lines_fld.clear();
        int64 start = getTickCount();
        // Detect the lines with FLD
        fld->detect(image, lines_fld);
        double duration_ms = double(getTickCount() - start) * 1000 / freq;
        //std::cout << "Elapsed time for FLD " << duration_ms << " ms." << std::endl;
    }
    // Show found lines with FLD
    Mat line_image_fld(image);
    fld->drawSegments(line_image_fld, lines_fld);
    imshow("FLD result", line_image_fld);

#endif








    //![edge_detection]
    // Edge detection

#if 0
    const auto kernel_size = 5;
    Mat src_gray;
    GaussianBlur(src, src_gray, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);

    Mat dst;
    Canny(src, dst, 50, 200, 3);
#endif

#if 0
    Mat dst;
    filter2DFreq(src, dst);
    normalize(dst, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8U);

    const auto borderSize = 20;

    cv::Rect roi(borderSize, borderSize, dst.cols - borderSize * 2, dst.rows - borderSize * 2);
    dst = dst(roi);

    const auto filtered = dst.clone();

    //Canny(dst, dst, 50, 200, 3);

    //auto dst = src;
#endif

#if 0
    Mat flt;
    filter2DFreq(src, flt);
    //normalize(dst, dst, 0, 255, NORM_MINMAX);
    //dst.convertTo(dst, CV_8U);

    const auto borderSize = 20;

    cv::Rect roi(borderSize, borderSize, flt.cols - borderSize * 2, flt.rows - borderSize * 2);
    flt = flt(roi);

    //const double alpha = 8.;
    //const double beta = -512.;
    const double alpha = 2.;
    const double beta = 128;

    Mat dst = Mat::zeros(flt.size(), src.type());// = src.clone();

    for (int y = 0; y < flt.rows; y++) {
        for (int x = 0; x < flt.cols; x++) {
            dst.at<uchar>(y, x) =
                saturate_cast<uchar>(alpha*flt.at<float>(y, x) + beta);
        }
    }

    const auto filtered = dst.clone();

    //Canny(dst, dst, 50, 200, 3);

    //auto dst = src;
#endif

#if 1
    //Mat dst = Mat::zeros(src.size(), src.type());// = src.clone();

    //const double alpha = 2.;
    //const double beta = 20.;

    //for (int y = 0; y < src.rows; y++) {
    //    for (int x = 0; x < src.cols; x++) {
    //            dst.at<uchar>(y, x) =
    //                saturate_cast<uchar>(alpha*src.at<uchar>(y, x) + beta);
    //    }
    //}
    //const auto kernel_size = 5;
    //GaussianBlur(dst, dst, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);

    //Mat dst;

    // heuristics

    Mat dst = src.clone();

    //for (int i = 1; i < std::min(dst.cols, dst.rows); ++i)
    //{
    //    const int threshold = 50;

    //    const auto start_value = dst.at<uchar>(0, i);
    //    for (int j = i; --j >= 0;)
    //    {
    //        const auto value = dst.at<uchar>(i - j - 1, j);
    //        if (value < start_value - threshold)
    //            break;
    //        dst.at<uchar>(i - j - 1, j) = start_value;
    //    }
    //}


    Mat background;
    GaussianBlur(dst, background, Size(63, 63), 0, 0, BORDER_DEFAULT);
    background -= 1;

    const auto kernel_size = 3;
    GaussianBlur(dst, dst, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);
    const auto filtered = dst.clone();


    Mat stripeless;
    GaussianBlur(dst, stripeless, Size(63, 1), 0, 0, BORDER_DEFAULT);

    Mat func = dst - stripeless + 128;

    /*
    Mat funkyBackground;
    GaussianBlur(func, funkyBackground, Size(31, 31), 0, 0, BORDER_DEFAULT);
    funkyBackground -= 1;

    Mat funkyDiff = func < funkyBackground;
    //*/

    Mat funkyFunc;
    adaptiveThreshold(func, funkyFunc, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1.);

    //Mat funkyDiff = filtered < background;


    Mat diff = filtered < background;

    // mask
    Mat mask;
    threshold(dst, mask, 180, 255, THRESH_BINARY_INV);
    int horizontal_size = 40;// dst.rows / 30;
    // Create structure element for extracting vertical lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
    // Apply morphology operations
    //erode(dst, dst, verticalStructure);
    dilate(mask, mask, horizontalStructure);

    int circular_size = 5;// dst.rows / 30;
    Mat circularStructure = getStructuringElement(MORPH_ELLIPSE, Size(circular_size, circular_size));
    dilate(mask, mask, circularStructure);

#endif

    //{
    //    auto neg = 255 - dst;
    //    Mat buf1;
    //    adaptiveThreshold(neg, buf1, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, 0);
    //    Mat buf2;
    //    adaptiveThreshold(dst, buf2, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 2);
    //    buf1 += buf2;
    //    threshold(buf1, dst, 1, 255, THRESH_BINARY);
    //}

    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 2);

    //Mat accum;
    //bool first = true;
    //for (int i = 11; i <= 63; i += 2)
    //{
    //    Mat buf;
    //    adaptiveThreshold(dst, buf, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, i, 2);
    //    if (first)
    //        accum = buf;
    //    else
    //        accum += buf;

    //    first = false;
    //}

    //threshold(accum, dst, 16, 255, THRESH_BINARY);

    //imshow("Transform", dst);
    //waitKey();

    dst &= mask;

    dst &= diff;

    dst &= funkyFunc;

    //medianBlur(dst, dst, 3);

    //bitwise_not(dst, dst);

    auto thresh = dst.clone();

    thinning(dst, dst);

    // Specify size on vertical axis
    int vertical_size = 4;// dst.rows / 30;
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
    vector<Vec4i> linesP; // will hold the results of the detection
    const int threshold = 10;
    HoughLinesP(dst, linesP, 1, CV_PI/180/10, threshold, 5, 25 ); // runs the actual detection
    //![hough_lines_p]
    //![draw_lines_p]
    // Draw the lines
    //for( size_t i = 0; i < linesP.size(); i++ )

    linesP.erase(std::remove_if(linesP.begin(), linesP.end(), [&dst](const Vec4i& l) {
        const double expectedAlgle = 0.05;
        const auto border = 10;
        return l[1] == l[3] || fabs(double(l[0] - l[2]) / (l[1] - l[3]) + expectedAlgle) > expectedAlgle
            || l[0] < border && l[2] < border || l[1] == 0 && l[3] == 0
            || l[0] >= (dst.cols - border) && l[2] >= (dst.cols - border) || l[1] == dst.rows - 1 && l[3] == dst.rows - 1;
    }), linesP.end());

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
        line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), color, 1, LINE_AA);
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

    reducedLines0.erase(std::remove_if(reducedLines0.begin(), reducedLines0.end(), [](const Vec4i& line) {
        return hypot(line[2] - line[0], line[3] - line[1]) <= 10;
    }), reducedLines0.end());

    auto reducedLines = reduceLines(reducedLines0, 50, 0.7, 2.5);

    //reducedLines.erase(std::remove_if(reducedLines.begin(), reducedLines.end(), [](const Vec4i& line) {
    //    return hypot(line[2] - line[0], line[3] - line[1]) <= 10;
    //}), reducedLines.end());

    Mat reducedLinesImg0 = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
    {
        RNG rng(215526);
        for (int i = 0; i < reducedLines0.size(); ++i)
        {
            auto color = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
            auto& reduced = reducedLines0[i];
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
        pointCloud.push_back(Point2i(reduced[0] - centerX, reduced[1] - centerY));
        pointCloud.push_back(Point2i(reduced[2] - centerX, reduced[3] - centerY));
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
#if 1
    for (int i = reducedLines.size(); --i >= 0;)
    {
        auto& line = reducedLines[i];
        if (hypot(line[2] - line[0], line[3] - line[1]) > 40)
            continue;

        auto val = sortLam(line);

        double dist;
        decltype(reducedLines)::iterator it;
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
        for (auto &detectedLine : { line , *it}) {
            pointCloud.push_back(Point2i(detectedLine[0], detectedLine[1]));
            pointCloud.push_back(Point2i(detectedLine[2], detectedLine[3]));
        }

        line = HandlePointCloud(pointCloud);

        reducedLines.erase(it);
    }
#endif

    // normalize direction
    for (auto& line : reducedLines) {
        if (line[1] > line[3]) {
            std::swap(line[0], line[2]);
            std::swap(line[1], line[3]);
        }
    }

    Mat reducedLinesImg = Mat::zeros(dst.rows, dst.cols, CV_8UC3);
    {
        RNG rng(215526);
        for (int i = 0; i < reducedLines.size(); ++i)
        {
            auto color = Scalar(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));;
            auto& reduced = reducedLines[i];
            line(reducedLinesImg, Point(reduced[0], reduced[1]), Point(reduced[2], reduced[3]), color, 2);
        }
    }

    for (int i = 0; i < reducedLines.size() - 1; ++i) {
        auto& first = reducedLines[i];
        auto& second = reducedLines[i + 1];
        if (first[1] > second[1])
            continue;

        const auto y_first = first[0] * sin_phi + first[1] * cos_phi;
        const auto y_second = second[0] * sin_phi + second[1] * cos_phi;

        const auto x_first = first[0] * cos_phi - first[1] * sin_phi;
        const auto x_second = second[0] * cos_phi - second[1] * sin_phi;

        const auto diff = y_second - y_first;

        std::cout << (diff * 1.27) << '\n';
    }


    //![imshow]
    // Show results
    imshow("Source", src);

    imshow("Background", background);

    imshow("Diff", diff);


    imshow("Stripeless", stripeless);

    imshow("Func", func);

    imshow("funkyFunc", funkyFunc);


    imshow("Filtered", filtered);

    imshow("Mask", mask);

    imshow("Transform", dst);

    imshow("Threshold", thresh);

    //imshow("Detected Lines (in red) - Line Transform", cdst);

    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    //![imshow]


    //imshow("Detected Lines", detectedLinesImg);
    imshow("Reduced Lines 0", reducedLinesImg0);
    imshow("Reduced Lines", reducedLinesImg);


    //![exit]
    // Wait and Exit
    waitKey();
    return 0;
    //![exit]
}
