# detect-lines

![alt text](images/IMG_1427.JPG)

* Image reading
* Image low pass filtering
* Emphasis on low image intensities:
* Background subtraction, horizontal stripes removal

* Obtaining one-dimensional (special case) Fourier spectra for image fragments using https://github.com/leerichardson/tree-swdft-2D
* Finding the separation line points where spectra main frequencies abruptly change
* Preliminary filtering of points using cv::partition and nanoflann library 
* Using RANSAC to find the separation line using separation line points
* Using CERES solver to refine the separation line

* Using both adaptiveThreshold and binary background subtraction to generate threshold image
* Using thinning to obtain lines skeletons
* Calling erode/dilate to filter out vertical lines
* Invoking HoughLinesP to generate lines from skeletons
* Merging lines according to https://stackoverflow.com/a/51121483/10472202
* Filtering out short lines
* Cutting lines using the RANSAC/CERES separation line mentioned above
* Filtering out short lines once more

* Search for the beginnings of short stripes with "known good" SURF data
* Movement to the left and up along long lines

* Merging HoughLinesP and SURF results to obtain the final data

![alt text](results/borderline0.jpg)
![alt text](results/Detected_Lines_(in_red)_-_Probabilistic_Line_Transform.jpg)
![alt text](results/Diff.jpg)
![alt text](results/Dst_before.jpg)
![alt text](results/func.jpg)
![alt text](results/image.jpg)
![alt text](results/imgCoherencyBin.jpg)
![alt text](results/imgOrientationBin.jpg)
![alt text](results/Mask.jpg)
![alt text](results/outSkeleton.jpg)
![alt text](results/qrwer.jpg)
![alt text](results/Reduced_Lines.jpg)
![alt text](results/Reduced_Lines_0.jpg)
![alt text](results/theMask.jpg)
![alt text](results/Thinning.jpg)
![alt text](results/Transform.jpg)
