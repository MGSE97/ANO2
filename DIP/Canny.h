#pragma once

#include "Method.h"

class CannyED : public Method
{
public:
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override {};

	bool predict(int i, cv::Mat& input, cv::Mat& output) override
	{
		const auto size = input.rows * input.cols;
		cv::Mat tmp, tmp2;
		
		// Blur noise
		blur(input, tmp, cv::Size(5, 5));

		// Canny edge detection
		//Canny(tmp, tmp, 110, 256); //150, 255
		AutoCanny(tmp, tmp2);

		// Blur noise
		blur(tmp2, tmp, cv::Size(4, 4));
		output = tmp;

		//threshold(tmp, tmp, 50, 255, THRESH_BINARY);

		const auto positive_count = cv::countNonZero(tmp);
		const auto predict_label = positive_count > (size / 14); // 20

		putText(output, std::to_string(positive_count), cv::Point(3, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 0, 0, 0));
		putText(output, std::to_string(i) + " " + std::to_string(size / 14), cv::Point(3, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0, 0));
		
		return predict_label;
	}

	// https://answers.opencv.org/question/176494/how-to-calculate-the-median-of-pixel-values-in-opencv-python/?sort=oldest
	int static Median(cv::Mat* img, int nVals) {
		// COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
		float range[] = { 0, nVals };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		int channels[] = { 0 };
		cv::Mat hist;
		calcHist(img, 1, channels, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);

		// COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
		cv::Mat cdf;
		hist.copyTo(cdf);
		for (int i = 1; i < nVals - 1; i++) {
			cdf.at<float>(i) += cdf.at<float>(i - 1);
		}
		cdf /= static_cast<float>(img->total());

		// COMPUTE MEDIAN
		int medianVal = 0;
		for (int i = 0; i < nVals - 1; i++) {
			//printf("%f\n", cdf.at<float>(i));
			if (cdf.at<float>(i) >= 0.5) { medianVal = i;  break; }
		}
		//printf("%d\n", medianVal);

		cdf.deallocate();
		hist.deallocate();

		return medianVal;
	}

	// https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
	void static AutoCanny(cv::Mat& src, cv::Mat& edges, float sigma = 0.33)
	{
		//compute the median of the single channel pixel intensities
		const auto v = Median(&src, 256);

		//apply automatic Canny edge detection using the computed median
		const auto lower = int(MAX(0, (1.0 - sigma) * v));
		const auto upper = int(MIN(255, (1.0 + sigma) * v));
		//printf("%d, %d\n", lower, upper);
		cv::Canny(src, edges, lower, upper);
	}
};


