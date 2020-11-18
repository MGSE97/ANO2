#pragma once

#include "Method.h"
#include "Canny.h"

#define M_PI 3.14159265359
#define IMG_Type unsigned char

class Threshold : public Method
{
public:
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override {};

	bool predict(int i, cv::Mat& input, cv::Mat& output) override
	{
		const auto size = input.rows * input.cols;
		cv::Mat tmp, tmp2, tmp4, tmp3 = cv::Mat::zeros(input.rows, input.cols, CV_8U);

		// Blur noise
		blur(input, tmp, cv::Size(5, 5));
		//blur(tmp, tmp2, Size(5, 5));
		AutoTreshold(tmp, tmp, 0.19, cv::THRESH_BINARY);

		// Normalize using Local Binary Patterns
		//LBP(*plot, &tmp3, 8, 64);
		LBP(tmp, &tmp3, std::min(input.rows, input.cols) / 8, 16);

		// Blur noise
		blur(tmp3, tmp, cv::Size(15, 15));
		output = tmp;

		const auto positive_count = countNonZero(tmp);
		const auto black_count = size - positive_count;
		const double black_white_ratio = (double)positive_count / (double)size;
		//printf("%f\n", black_white_ratio);
		const auto predict_label = black_white_ratio > 0.4; // 20

		putText(output, std::to_string(black_white_ratio).substr(2, 4), cv::Point(3, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 0, 0, 0));
		//putText(tmp, to_string(black_count), Point(3, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 0, 0, 0));
		putText(output, std::to_string(i), cv::Point(3, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0, 0));

		return predict_label;
	};

	void static AutoTreshold(cv::Mat& src, cv::Mat& dst, float sigma, int method)
	{
		//compute the median of the single channel pixel intensities
		const auto v = CannyED::Median(&src, 256);

		const auto lower = int(MAX(0, (1.0 - sigma) * v));
		const auto upper = int(MIN(255, (1.0 + sigma) * v));
		//printf("%d, %d [%d]\n", lower, upper, v);
		threshold(src, dst, lower, upper, method);
	}

	bool static is_in_range(cv::Mat& src, int r, int c)
	{
		return r >= 0 && r < src.rows&& c >= 0 && c < src.cols;
	}

	// https://www.bytefish.de/blog/local_binary_patterns.html
	void static LBP(cv::Mat& src, cv::Mat* dst, int radius, int neighbors)
	{
		/*for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				const auto center = src.at<IMG_Type>(i, j);
				unsigned char code = 0;

				if (is_in_range(src, i - 1, j - 1)) code |= (src.at<IMG_Type>(i - 1, j - 1) > center) << 7;
				if (is_in_range(src, i - 1, j)) code |= (src.at<IMG_Type>(i - 1, j) > center) << 6;
				if (is_in_range(src, i - 1, j + 1)) code |= (src.at<IMG_Type>(i - 1, j + 1) > center) << 5;
				if (is_in_range(src, i, j + 1)) code |= (src.at<IMG_Type>(i, j + 1) > center) << 4;
				if (is_in_range(src, i + 1, j + 1)) code |= (src.at<IMG_Type>(i + 1, j + 1) > center) << 3;
				if (is_in_range(src, i + 1, j)) code |= (src.at<IMG_Type>(i + 1, j) > center) << 2;
				if (is_in_range(src, i + 1, j - 1)) code |= (src.at<IMG_Type>(i + 1, j - 1) > center) << 1;
				if (is_in_range(src, i, j - 1)) code |= (src.at<IMG_Type>(i, j - 1) > center) << 0;

				dst.at<IMG_Type>(i, j) = code;
			}
		}*/

		neighbors = std::max(std::min(neighbors, 31), 1);
		*dst = cv::Mat::zeros(src.rows, src.cols, CV_8U);

		for (int n = 0; n < neighbors; n++) {
			// sample points
			float x = static_cast<float>(radius) * cos(2.0 * M_PI * n / static_cast<float>(neighbors));
			float y = static_cast<float>(radius) * -sin(2.0 * M_PI * n / static_cast<float>(neighbors));
			// relative indices
			int fx = static_cast<int>(floor(x));
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));
			int cy = static_cast<int>(ceil(y));
			// fractional part
			float ty = y - fy;
			float tx = x - fx;
			// set interpolation weights
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx * (1 - ty);
			float w3 = (1 - tx) * ty;
			float w4 = tx * ty;
			// iterate through your data
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					float t = w1 * is_in_range(src, i + fy, j + fx) ? src.at<IMG_Type>(i + fy, j + fx) : 0
						+ w2 * is_in_range(src, i + fy, j + fx) ? src.at<IMG_Type>(i + fy, j + cx) : 0
						+ w3 * is_in_range(src, i + fy, j + fx) ? src.at<IMG_Type>(i + cy, j + fx) : 0
						+ w4 * is_in_range(src, i + fy, j + fx) ? src.at<IMG_Type>(i + cy, j + cx) : 0;
					// we are dealing with floating point precision, so add some little tolerance
					dst->at<IMG_Type>(i, j) += ((t > src.at<IMG_Type>(i, j)) && (abs(t - src.at<IMG_Type>(i, j)) > std::numeric_limits<float>::epsilon())) << n;
				}
			}
		}
	}
};
