#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/ximgproc.hpp"

#include "opencv2/xfeatures2d.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <fstream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
    const char* filename = argv[1];
    // Loads an image
    Mat src = imread(filename, IMREAD_GRAYSCALE);

    Mat lowLevelFloat = Mat::zeros(src.size(), CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int v = src.at<uchar>(y, x);
            lowLevelFloat.at<float>(y, x) = std::clamp(v, 0, 1) * 10;
        }
    }

    /*
    enum { SIZE = 800 };
    resize(src, src, Size(SIZE, SIZE), 0, 0, INTER_LANCZOS4);
    resize(lowLevelFloat, lowLevelFloat, Size(SIZE, SIZE), 0, 0, INTER_LANCZOS4);
    */

    GaussianBlur(lowLevelFloat, lowLevelFloat, Size(9, 35), 0, 0, BORDER_DEFAULT);
    Mat lowLevel;
    lowLevelFloat.convertTo(lowLevel, CV_8U);

    src += lowLevel;


    Mat dst = src.clone();


    const auto kernel_size = 3;
    GaussianBlur(dst, dst, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);
    const auto filtered = dst.clone();

    //
    Mat dstFloat;
    src.convertTo(dstFloat, CV_32F);

    Mat backgroundFloat;
    GaussianBlur(dstFloat, backgroundFloat, Size(63, 63), 0, 0, BORDER_DEFAULT);
    backgroundFloat -= 0.5;

    Mat background;
    backgroundFloat.convertTo(background, CV_8U);

    GaussianBlur(dstFloat, dstFloat, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);

    Mat diff = dstFloat < backgroundFloat;


    Mat stripeless;
    GaussianBlur(dstFloat, stripeless, Size(63, 1), 0, 0, BORDER_DEFAULT);

    Mat funcFloat = (dstFloat - stripeless + 32.) * 4.;
    //GaussianBlur(funcFloat, funcFloat, Size(3, 3), 0, 0, BORDER_DEFAULT);
    Mat func;
    funcFloat.convertTo(func, CV_8U);

    /*
    imshow("Func", func);

    waitKey();
    */

    imwrite(argv[2], func);

    return 0;

}
