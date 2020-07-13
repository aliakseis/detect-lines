/**
 * @file houghclines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/ximgproc.hpp"


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
    Mat dst = Mat::zeros(src.size(), src.type());// = src.clone();

    const double alpha = 2.;
    const double beta = 20.;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
                dst.at<uchar>(y, x) =
                    saturate_cast<uchar>(alpha*src.at<uchar>(y, x) + beta);
        }
    }
    const auto kernel_size = 5;
    GaussianBlur(dst, dst, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);
    const auto filtered = dst.clone();
#endif
    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);

    medianBlur(dst, dst, 3);

    bitwise_not(dst, dst);

    thinning(dst, dst);

    // Specify size on vertical axis
    int vertical_size = 2;// dst.rows / 30;
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    // Apply morphology operations
    erode(dst, dst, verticalStructure, Point(-1, -1));
    dilate(dst, dst, verticalStructure, Point(-1, -1));


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
    HoughLinesP(dst, linesP, 1, CV_PI/180/10, 15, 5, 15 ); // runs the actual detection
    //![hough_lines_p]
    //![draw_lines_p]
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        if (l[1] == l[3])
            continue;
        if (fabs(l[0] - l[2]) / fabs(l[1] - l[3]) > 0.1)
            continue;

        auto color = (min(l[1], l[3]) < 380) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), color, 1, LINE_AA);
    }
    //![draw_lines_p]

    //![imshow]
    // Show results
    imshow("Source", src);

    imshow("Filtered", filtered);

    imshow("Transform", dst);

    //imshow("Detected Lines (in red) - Line Transform", cdst);

    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    //![imshow]

    //![exit]
    // Wait and Exit
    waitKey();
    return 0;
    //![exit]
}
