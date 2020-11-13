# ANO2

Image Analysis II. Detection of parking lot availablity using multiple methods.


## Getting Started

1. Download this repository
2. Unzip [OpenCV](opencv.zip) into solution folder
3. Install [CUDA Toolkit 10.2 or newer](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
4. Download and unzip [DLib](http://dlib.net/compile.html)
    - Open console
    - Navigate to DLib folder
    - Build DLib
      ```
      mkdir build
      cd build
      cmake -G "Visual Studio 16 2019" -T host=x64 ..
      ```
    - Install Dlib, by default, it will create includes and library in `%Program Files%\dlib_project`
      ```
      cmake --build . --config Release --target INSTALL
      ```
5. Open [solution](DIP.sln)  
6. Go to DIP project properties and check Library locations to match within your system
7. Merge [config.h](DIP/config.h) file with DLib builded in `dlib-19.21\build\dlib\config.h`
8. Build & Run

### Prerequisites

Software:
* [OpenCV](https://opencv.org/) *included* - Image manipulation library
* [DLib](http://dlib.net/) - Neural Networks API
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - CUDA GPU Acceleration, used in DLib
* [CMake](https://cmake.org/) - Building tool, used for DLib

IDE:
* [Visual Studio 2019](https://visualstudio.microsoft.com/cs/vs/)

### Solution

Only one project. C++

## Description

Program will show final detected image and parking lots processed images for each image in [**test_images** folder](DIP/test_images).
After all images are evaluated, it will print statistics to console.

![GUI visualization](Resources/M_Combo.png)

### Methods

Methods can be enabled/disabled using **define** statements in code.
Only one method can be used at once.

### Without learning

|Method|Accuracy|F1 Score|Weaknesses|Info|
|------|:------:|:------:|----------|----|
|Canny edge detection|99.2%|98.7%|Shadows, Far objects, Wide cars from near lots||
|Treshold, Local binnary patterns|98.7%|97.9%|Noise from ground|Wrong usage|

### Partial learning

|Method|Accuracy|F1 Score|Learning time|Weaknesses|Info|
|------|:------:|:------:|:-----------:|----------|----|
|LBP, HOG, Comparison Day/Night|80.4%|70.7%|Short|Slow|Loads free lots images|

### With learning

|Method|Accuracy|F1 Score|Learning time|Epochs|Batch Size|Learning Rate|Weaknesses|Info / Loss|
|------|:------:|:------:|:-----------:|:----:|:--------:|:-----------:|----------|-----------|
|LBP, HOG, SVM|70.5%|27.4%|Short||||Weak predictions||
||||||||||
|CNN XS|92.3%|89.1%|Short|100|512|1e-2|Small network, Shadows, Night|From lecture, DLib|
||||||||||
|Alex Net|-|-|Long|-|-|-|Large, Sensitive|DLib| 
|->|98.9%|98.3%|30 min|200|128|1e-5||~0.000872731|                      
|->|98.7%|98.0%|15 min|90|32|1e-5||~0.0114783|
|->|98.7%|97.9%|1.5 h|398|32|1e-6||~0.0487347|
||||||||||
|Combination|99.3%|98.9%|Slow||||Loads free lots images|Combination of above|

## Author

* [**MGSE97**](https://github.com/MGSE97)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

This project uses OpenCV library, see the [License](OpenCV-License.txt) file for details.
