# ANO2

Image Analysis II. Detection of parking lot availablity using multiple methods.


## Getting Started

1. Download this repository
2. Unzip [opencv.zip] into solution folder
3. Open [solution](DIP.sln)  
4. Build & Run

### Prerequisites

Software:
* [OpenCV](https://opencv.org/) *included* - Image manipulation library

IDE:
* [Visual Studio 2019](https://visualstudio.microsoft.com/cs/vs/)

### Solution

Only one project. C++

## Description

Program will show final detected image and parking lots processed images for each image in [**test_images** folder](DIP/test_images).
After all images are evaluated, it will print statistics to console.

### Methods

Methods can be enabled/disabled using **define** statements in code.
Only one method can be used at once.

### Without Nerual Networks

|Method|F1 Score|Weaknesses|Info|
|------|--------|----------|----|
|Canny edge detection|98.7%|Shadows, Far objects, Wide cars from near lots||
|Treshold, Local binnary patterns|97.9%|Noise from ground|Wrong usage|

### With Neural Networks

|Method|F1 Score|Info|
|------|--------|----|

## Author

* [**MGSE97**](https://github.com/MGSE97)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

This project uses OpenCV library, see the [License](OpenCV-License.txt) file for details.