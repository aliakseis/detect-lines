#include "detect-lines.h"

//#include "known-good.h"

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
    // Declare the output variables
    //Mat dst, cdst, cdstP;

    //![load]
    const char* default_file = "../data/sudoku.png";
    const char* filename = argc >= 2 ? argv[1] : default_file;

    const char* outPath = argc >= 3 ? argv[2] : ".";

    Mat src = imread(filename);// , IMREAD_GRAYSCALE);

    auto lam = [outPath](std::string name, cv::InputArray img) {
        std::replace(name.begin(), name.end(), ' ', '_');
        imwrite(outPath + ('/' + name) + ".jpg", img);
    };

    auto lines = imaging::calculating(filename, lam);

    Scalar color = Scalar(0, 255, 0);
    int radius = 2;
    int thickness = -1;
    for (auto& line : lines) {
        circle(src, Point(std::get<0>(line), std::get<1>(line)),  radius, color, thickness);
        circle(src, Point(std::get<2>(line), std::get<3>(line)), radius, color, thickness);
    }

    imshow("Reduced Lines", src);

    waitKey();
}
