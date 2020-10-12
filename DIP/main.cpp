#include "stdafx.h"
#include <iostream>

//opencv - https://opencv.org/
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;

// dlib alternativa
//dlib - http://dlib.net/
/*#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
using namespace dlib;
*/

struct space
{
	int x01, y01, x02, y02, x03, y03, x04, y04, occup;
};

int load_parking_geometry(const char* filename, space* spaces);
void extract_space(space* spaces, Mat in_mat, std::vector<Mat>& vector_images);
void draw_detection(space* spaces, Mat& frame);
void evaluation(fstream& detectorOutputFile, fstream& groundTruthFile);

void train_parking();
void test_parking();

void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData);

int spaces_num = 56;
cv::Size space_size(80, 80);

#define USE_NN false
#define WAIT 500

int main(int argc, char** argv)
{
	if (USE_NN)
	{
		cout << "Train OpenCV Start" << endl;
		train_parking();
		cout << "Train OpenCV End" << endl;
	}

	cout << "Test OpenCV Start" << endl;
	test_parking();
	cout << "Test OpenCV End" << endl;

}

void train_parking()
{
	//load parking lot geometry
	space* spaces = new space[spaces_num];
	load_parking_geometry("parking_map.txt", spaces);

	std::vector<Mat> train_images;
	std::vector<int> train_labels;

	fstream train_file("train_images.txt");
	string train_path;

	while (train_file >> train_path)
	{

		//cout << "train_path: " << train_path << endl;
		Mat frame;

		//read training images
		frame = imread(train_path, 0);

		// label = 1;//occupied place
		// label = 0;//free place         
		int label = 0;
		if (train_path.find("full") != std::string::npos) label = 1;

		//extract each parking space
		extract_space(spaces, frame, train_images);

		//training label for each parking space
		for (int i = 0; i < spaces_num; i++)
		{
			train_labels.push_back(label);
		}

	}

	delete spaces;

	cout << "Train images: " << train_images.size() << endl;
	cout << "Train labels: " << train_labels.size() << endl;

	//TODO - Train

}

bool is_in_range(Mat& src, int r, int c)
{
	return r >= 0 && r < src.rows&& c >= 0 && c < src.cols;
}

#define M_PI 3.14159265359
#define IMG_Type unsigned char
// https://www.bytefish.de/blog/local_binary_patterns.html
void LBP(Mat& src, Mat* dst, int radius, int neighbors)
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

	neighbors = max(min(neighbors, 31), 1);
	*dst = Mat::zeros(src.rows, src.cols, CV_8U);

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

// https://answers.opencv.org/question/176494/how-to-calculate-the-median-of-pixel-values-in-opencv-python/?sort=oldest
int Median(cv::Mat* img, int nVals) {
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
void AutoCanny(Mat& src, Mat& edges, float sigma = 0.33)
{
	//compute the median of the single channel pixel intensities
	const auto v = Median(&src, 256);

	//apply automatic Canny edge detection using the computed median
	const auto lower = int(MAX(0, (1.0 - sigma) * v));
	const auto upper = int(MIN(255, (1.0 + sigma) * v));
	//printf("%d, %d\n", lower, upper);
	Canny(src, edges, lower, upper);
}

void AutoTreshold(Mat& src, Mat& dst, float sigma = 0.33)
{
	//compute the median of the single channel pixel intensities
	const auto v = Median(&src, 256);

	const auto lower = int(MAX(0, (1.0 - sigma) * v));
	const auto upper = int(MIN(255, (1.0 + sigma) * v));
	//printf("%d, %d\n", lower, upper);
	
	threshold(src, dst, lower, upper, THRESH_BINARY);
}

#define M_CANNY 0
#define M_LBP 1
#define NON_NN_METHOD M_LBP

bool is_occupied(int i, Mat* plot, std::vector<Mat>* spaces_imgs, Mat* spaces_img, Mat* local_spaces_img)
{
	const auto size = plot->rows * plot->cols;
	if (USE_NN)
	{
		return true;
	}
	else
	{
#if NON_NN_METHOD == M_CANNY
		Mat tmp, tmp2;

		// Blur noise
		blur(*plot, tmp, Size(5, 5));

		// Canny edge detection
		//Canny(tmp, tmp, 110, 256); //150, 255
		AutoCanny(tmp, tmp2);

		// Blur noise
		blur(tmp2, tmp, Size(3, 3));

		//threshold(tmp, tmp, 50, 255, THRESH_BINARY);

		const auto positive_count = countNonZero(tmp);
		const auto predict_label = positive_count > (size / 18); // 20
		
#elif NON_NN_METHOD == M_LBP
		Mat tmp, tmp2, tmp3 = Mat::zeros(plot->rows, plot->cols, CV_8U);

		// Blur noise
		blur(*plot, tmp, Size(5, 5));
		//blur(tmp, tmp2, Size(5, 5));
		AutoTreshold(tmp, tmp, 0.19);

		// Normalize using Local Binary Patterns
		//LBP(*plot, &tmp3, 8, 64);
		LBP(tmp, &tmp3, min(plot->rows, plot->cols)/8, 16);

		/*int channels[] = { 0, 0 };
		for (int i = 4; i <= 16; i+=2)
		{
			LBP(tmp, &tmp2, i, 16);
			Mat in[] = { tmp2, tmp3 };
			mixChannels(in, 1, &tmp3, 1, channels, 1);
		}*/
		//tmp = tmp2;

		// Blur noise
		blur(tmp3, tmp, Size(15, 15));
		//tmp = tmp3;
		
		// Canny edge detection
		//Canny(tmp, tmp, 110, 256); //150, 255
		//AutoCanny(tmp2, tmp);
		//LBP(tmp, &tmp2, 4, 16);
		//blur(tmp2, tmp, Size(10, 10));
		//AutoCanny(tmp, tmp2);
		//AutoCanny(*plot, tmp);
		//tmp = tmp2;

		// Blur noise
		//blur(tmp2, tmp, Size(3, 3));

		//threshold(tmp, tmp, 50, 255, THRESH_BINARY);
		//AutoTreshold(tmp, tmp);

		const auto positive_count = countNonZero(tmp);
		const double black_white_ratio = (double)positive_count / (double)size;
		//printf("%f\n", black_white_ratio);
		const auto predict_label = black_white_ratio > 0.4; // 20
		//const auto predict_label = (size - positive_count) > 100; // 20
#endif

		// Display grid magic
		vconcat(*plot, tmp, *local_spaces_img);
		if (i == 0)
			local_spaces_img->copyTo(*spaces_img);
		else if (i % 15 == 0)
		{
			spaces_imgs->push_back(*spaces_img);
			local_spaces_img->copyTo(*spaces_img);
		}
		else
			hconcat(*spaces_img, *local_spaces_img, *spaces_img);

		return predict_label;
	}
}

/**
 * \brief
 */
void test_parking()
{

	space* spaces = new space[spaces_num];
	load_parking_geometry("parking_map.txt", spaces);

	fstream test_file("test_images.txt");
	ofstream out_label_file("out_prediction.txt");
	string test_path;

	//namedWindow("test_img", 0);
	//namedWindow("edges_img", 0);
	namedWindow("spaces_img", 0);

	while (test_file >> test_path)
	{
		//cout << "test_path: " << test_path << endl;
		Mat frame, gradX, gradY;
		//read testing images
		frame = imread(test_path, 1);
		Mat draw_frame = frame.clone();
		cvtColor(frame, frame, COLOR_BGR2GRAY);

		std::vector<Mat> test_images;
		extract_space(spaces, frame, test_images);

		std::vector<Mat> spaces_imgs;
		Mat edges, test_image, spaces_img, local_spaces_img;
		int colNum = 0, predict_label = 0;
		for (int i = 0; i < test_images.size(); i++)
		{
			predict_label = is_occupied(i, &test_images[i], &spaces_imgs, &local_spaces_img, &spaces_img);

			out_label_file << predict_label << endl;
			spaces[i].occup = predict_label;
		}

		// Display grid magic, missing record
		int space_images = spaces_imgs.size() - 1;
		if (spaces_imgs[space_images].data != spaces_img.data)
			spaces_imgs.push_back(spaces_img);

		// Display grid magic, row merge
		for (int i = space_images; i >= 0; i--)
		{
			if (i == space_images)
				spaces_img = spaces_imgs[i];
			else
				vconcat(spaces_img, spaces_imgs[i], spaces_img);
		}

		//draw detection
		draw_detection(spaces, draw_frame);
		namedWindow("draw_frame", 0);
		imshow("draw_frame", draw_frame);
		imshow("spaces_img", spaces_img);
		waitKey(WAIT);
	}



	//evaluation    
	fstream detector_file("out_prediction.txt");
	fstream groundtruth_file("groundtruth.txt");
	evaluation(detector_file, groundtruth_file);
}

int load_parking_geometry(const char* filename, space* spaces)
{
	FILE* file = fopen(filename, "r");
	if (file == NULL) return -1;
	int ps_count, i, count;
	count = fscanf(file, "%d\n", &ps_count); // read count of polygons
	for (i = 0; i < ps_count; i++) {
		int p = 0;
		int poly_size;
		count = fscanf(file, "%d->", &poly_size); // read count of polygon vertexes
		int* row = new int[poly_size * 2];
		int j;
		for (j = 0; j < poly_size; j++) {
			int x, y;
			count = fscanf(file, "%d,%d;", &x, &y); // read vertex coordinates
			row[p] = x;
			row[p + 1] = y;
			p = p + 2;
		}
		spaces[i].x01 = row[0];
		spaces[i].y01 = row[1];
		spaces[i].x02 = row[2];
		spaces[i].y02 = row[3];
		spaces[i].x03 = row[4];
		spaces[i].y03 = row[5];
		spaces[i].x04 = row[6];
		spaces[i].y04 = row[7];
		//printf("}\n");
		free(row);
		count = fscanf(file, "\n"); // read end of line
	}
	fclose(file);
	return 1;
}

void extract_space(space* spaces, Mat in_mat, std::vector<Mat>& vector_images)
{
	for (int x = 0; x < spaces_num; x++)
	{
		Mat src_mat(4, 2, CV_32F);
		Mat out_mat(space_size, CV_8U, 1);
		src_mat.at<float>(0, 0) = spaces[x].x01;
		src_mat.at<float>(0, 1) = spaces[x].y01;
		src_mat.at<float>(1, 0) = spaces[x].x02;
		src_mat.at<float>(1, 1) = spaces[x].y02;
		src_mat.at<float>(2, 0) = spaces[x].x03;
		src_mat.at<float>(2, 1) = spaces[x].y03;
		src_mat.at<float>(3, 0) = spaces[x].x04;
		src_mat.at<float>(3, 1) = spaces[x].y04;

		Mat dest_mat(4, 2, CV_32F);
		dest_mat.at<float>(0, 0) = 0;
		dest_mat.at<float>(0, 1) = 0;
		dest_mat.at<float>(1, 0) = out_mat.cols;
		dest_mat.at<float>(1, 1) = 0;
		dest_mat.at<float>(2, 0) = out_mat.cols;
		dest_mat.at<float>(2, 1) = out_mat.rows;
		dest_mat.at<float>(3, 0) = 0;
		dest_mat.at<float>(3, 1) = out_mat.rows;

		Mat H = findHomography(src_mat, dest_mat, 0);
		warpPerspective(in_mat, out_mat, H, space_size);

		//imshow("out_mat", out_mat);
		//waitKey(0);

		vector_images.push_back(out_mat);
	}

}

void draw_detection(space* spaces, Mat& frame)
{
	int sx, sy;
	for (int i = 0; i < spaces_num; i++)
	{
		Point pt1, pt2;
		pt1.x = spaces[i].x01;
		pt1.y = spaces[i].y01;
		pt2.x = spaces[i].x03;
		pt2.y = spaces[i].y03;
		sx = (pt1.x + pt2.x) / 2;
		sy = (pt1.y + pt2.y) / 2;
		if (spaces[i].occup)
		{
			circle(frame, Point(sx, sy - 25), 12, CV_RGB(255, 0, 0), -1);
		}
		else
		{
			circle(frame, Point(sx, sy - 25), 12, CV_RGB(0, 255, 0), -1);
		}
	}
}

void evaluation(fstream& detectorOutputFile, fstream& groundTruthFile)
{
	int detectorLine, groundTruthLine;
	int falsePositives = 0;
	int falseNegatives = 0;
	int truePositives = 0;
	int trueNegatives = 0;
	while (true)
	{
		if (!(detectorOutputFile >> detectorLine)) break;
		groundTruthFile >> groundTruthLine;

		int detect = detectorLine;
		int ground = groundTruthLine;

		//false positives
		if ((detect == 1) && (ground == 0))
		{
			falsePositives++;
		}

		//false negatives
		if ((detect == 0) && (ground == 1))
		{
			falseNegatives++;
		}

		//true positives
		if ((detect == 1) && (ground == 1))
		{
			truePositives++;
		}

		//true negatives
		if ((detect == 0) && (ground == 0))
		{
			trueNegatives++;
		}

	}
	cout << "falsePositives " << falsePositives << endl;
	cout << "falseNegatives " << falseNegatives << endl;
	cout << "truePositives " << truePositives << endl;
	cout << "trueNegatives " << trueNegatives << endl;
	float acc = (float)(truePositives + trueNegatives) / (float)(truePositives + trueNegatives + falsePositives + falseNegatives);
	float f1 = (float)(truePositives / (truePositives + 0.5 * (falsePositives * falseNegatives)));
	cout << "Accuracy " << acc << endl;
	cout << "F1 score " << f1 << endl;
}

void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	std::vector< Mat >::const_iterator itr = train_samples.begin();
	std::vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}
