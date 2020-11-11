#define OMP_NUM_THREADS 12

#include <iostream>

//opencv - https://opencv.org/
#include <opencv2/opencv.hpp>

//dlib - http://dlib.net
#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace dlib;

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

int Median(cv::Mat* img, int nVals);
bool is_in_range(Mat& src, int r, int c);
void LBP(Mat& src, Mat* dst, int radius, int neighbors);
void AutoCanny(Mat& src, Mat& edges, float sigma = 0.33);
void AutoTreshold(Mat& src, Mat& dst, float sigma = 0.33, int method = THRESH_BINARY);
std::vector<float> HOG(Mat& src, Mat* dst, Size block, Size cell, int nBins = 9, bool dominant = false);

void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData);

#define M_PI 3.14159265359
#define IMG_Type unsigned char

int spaces_num = 56;
cv::Size space_size(80, 80);

#define USE_NN true
#define WAIT 1

#define M_COMBO -1
#define M_CANNY 0
#define M_LBP 1
#define M_HOG 2
#define NON_NN_METHOD M_CANNY
//#define NON_NN_METHOD M_LBP
//#define NON_NN_METHOD M_HOG
//#define NON_NN_METHOD M_COMBO

#define M_DLIB_ICHL 0
#define NN_METHOD M_DLIB_ICHL
 // I 80x80x3 = 19200, C 80x80x1 = 6400, H 80x80x1 = 6400, All 80x80x5 = 32000

#if USE_NN == true && NN_METHOD == M_DLIB_ICHL
#endif

//using ichl = loss_multiclass_log<
//	relu < fc < 2,
//	relu < fc < 4,
//	relu < fc < 8,
//	relu < fc < 16,
//	relu < fc < 32,
//	relu < fc < 64,
//	max_pool < 2, 2, 2, 2, relu < con < 128, 5, 5, 1, 1,
//	max_pool < 2, 2, 2, 2, relu < con < 128, 5, 5, 1, 1,
//	max_pool < 2, 2, 2, 2, relu < con < 256, 5, 5, 1, 1,
//	max_pool < 2, 2, 2, 2, relu < con < 256, 5, 5, 1, 1,
//	max_pool < 2, 2, 2, 2, relu < con < 512, 5, 5, 1, 1,
//	max_pool < 2, 2, 2, 2, relu < con < 512, 5, 5, 1, 1,
//	max_pool < 2, 2, 2, 2, relu < con < 1024, 5, 5, 1, 1,
//	max_pool < 2, 2, 2, 2, relu < con < 1024, 5, 5, 1, 1,
//	input < matrix < unsigned char
//	>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using ichl = dlib::loss_multiclass_log<
	dlib::fc<2,
	dlib::relu< dlib::fc<84,
	dlib::relu< dlib::fc<120,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 16, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 6, 5, 5, 1, 1,
	dlib::input< dlib::matrix<unsigned char>>
	>>>>>>>>>>>>;

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

	std::vector<matrix<unsigned char>> train_images;
	std::vector<unsigned long> train_labels;

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
		std::vector<Mat> images;
		extract_space(spaces, frame, images);
		for (int i = 0; i < images.size(); i++) {
			cv_image<unsigned char> cimg(images[i]);
			matrix<unsigned char> dlibimg = dlib::mat(cimg);
			train_images.push_back(dlibimg);
		}

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
	ichl net;
	dnn_trainer<ichl> trainer(net, sgd(), { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
	//dnn_trainer<ichl> trainer(net);
	trainer.set_learning_rate(0.01);
	trainer.set_min_learning_rate(0.0001);
	trainer.set_mini_batch_size(100);
	trainer.set_iterations_without_progress_threshold(1000);
	trainer.set_max_num_epochs(20);
	trainer.be_verbose();

	cout << "Training ..." << endl;
	
	trainer.train(train_images, train_labels);

	cout << "Saving ..." << endl;

	serialize("ichl.dat") << net;
}


bool is_close(int a, int b, float distance)
{
	return a < (b* (1.0 + distance)) && a >(b * (1.0 - distance));
}

bool is_occupied(int i, Mat* plot, std::vector<Mat>* spaces_imgs, Mat* spaces_img, Mat* local_spaces_img, Mat* free_plot_day, Mat* free_plot_night)
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
		blur(tmp2, tmp, Size(4, 4));

		//threshold(tmp, tmp, 50, 255, THRESH_BINARY);

		auto positive_count = countNonZero(tmp);
		auto predict_label = positive_count > (size / 14); // 20

		putText(tmp, to_string(positive_count), Point(3, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 0, 0, 0));
		putText(tmp, to_string(i) + " " + to_string(size / 14), Point(3, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(255, 0, 0, 0));

#elif NON_NN_METHOD == M_LBP
		Mat tmp, tmp2, tmp4, tmp3 = Mat::zeros(plot->rows, plot->cols, CV_8U);

		// Blur noise
		blur(*plot, tmp, Size(5, 5));
		//blur(tmp, tmp2, Size(5, 5));
		AutoTreshold(tmp, tmp, 0.19, THRESH_BINARY);

		// Normalize using Local Binary Patterns
		//LBP(*plot, &tmp3, 8, 64);
		LBP(tmp, &tmp3, min(plot->rows, plot->cols) / 8, 16);

		int channels[] = { 0, 0 };
		/*for (int i = 4; i <= 16; i+=2)
		{
			LBP(tmp, &tmp2, i, 16);
			Mat in[] = { tmp2, tmp3 };
			mixChannels(in, 1, &tmp3, 1, channels, 1);
		}*/
		//tmp = tmp2;

		// Blur noise
		blur(tmp3, tmp, Size(15, 15));
		//tmp = tmp3;


		//blur(*plot, tmp, Size(5, 5));

		//// Normalize using Local Binary Patterns
		////LBP(tmp, &tmp3, min(plot->rows, plot->cols) / 8, 4);
		//LBP(tmp, &tmp2, 1, 2);
		//LBP(tmp, &tmp3, 2, 4);
		//LBP(tmp, &tmp2, 3, 8);
		////tmp2 *= 16;
		////dilate(tmp2, tmp, getStructuringElement(MORPH_DILATE, Size(3, 3)));
		////dilate(tmp, tmp2, getStructuringElement(MORPH_DILATE, Size(3, 3)));
		//Mat in[] = { tmp2, tmp3, tmp4 };
		//mixChannels(in, 1, &tmp, 1, channels, 1);
		////blur(tmp, tmp2, Size(3, 3));

		////AutoCanny(tmp, tmp2);
		////boxFilter(tmp, tmp2, -1, Size(3, 3), Point(-1,-1), false);
		////bilateralFilter(tmp, tmp2, 9, 64, 64);
		//tmp = tmp2/64;
		///*auto s = Size(3, 3);
		//for (int i = 0; i < 2; i++)
		//{
		//	dilate(tmp, tmp2, getStructuringElement(MORPH_DILATE, s));
		//	boxFilter(tmp2, tmp3, -1, s);
		//	boxFilter(tmp3, tmp2, -1, s);
		//	erode(tmp2, tmp, getStructuringElement(MORPH_ERODE, s));
		//}*/
		////
		////medianBlur(tmp2, tmp3, 5);
		////bilateralFilter(tmp3, tmp2, 5, 64, 64);
		////AutoTreshold(tmp2, tmp2, 0.33, THRESH_BINARY_INV);
		////Canny(tmp2, tmp, 8, 128);
		////blur(tmp2, tmp, Size(5, 5));
		////tmp = tmp2;
		////tmp *= 8.0;

		////threshold(tmp, tmp, 1, 4, THRESH_TRUNC);

		////tmp *= 4;

		const auto positive_count = countNonZero(tmp);
		const auto black_count = size - positive_count;
		const double black_white_ratio = (double)positive_count / (double)size;
		//printf("%f\n", black_white_ratio);
		const auto predict_label = black_white_ratio > 0.4; // 20
		//const auto predict_label = black_count > 800;
		//const auto predict_label = black_white_ratio > 0.8; // 20
		//const auto predict_label = (size - positive_count) > 100; // 20
		//
		putText(tmp, to_string(black_white_ratio).substr(2, 4), Point(3, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 0, 0, 0));
		//putText(tmp, to_string(black_count), Point(3, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 0, 0, 0));
		putText(tmp, to_string(i), Point(3, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(255, 0, 0, 0));

#elif NON_NN_METHOD == M_HOG
		Mat tmp, tmp2, tmp3, tmp4;

		//auto plotSize = Size(plot->rows, plot->cols); //80x80
		auto block = Size(32, 32);
		auto cell = Size(16, 16);

		// Blur noise
		blur(*plot, tmp, Size(5, 5));
		LBP(tmp, &tmp2, min(plot->rows, plot->cols) / 8, 16);

		blur(*free_plot_day, tmp3, Size(5, 5));
		LBP(tmp3, &tmp4, min(free_plot_day->rows, free_plot_night->cols) / 8, 16);
		auto free_vectors_day = HOG(tmp4, &tmp3, block, cell, 9, false);

		blur(*free_plot_night, tmp3, Size(5, 5));
		LBP(tmp3, &tmp4, min(free_plot_night->rows, free_plot_night->cols) / 8, 16);
		auto free_vectors_night = HOG(tmp4, &tmp3, block, cell, 9, false);


		auto vectors = HOG(tmp2, &tmp, block, cell, 9, false);
		float loss_day = 0.0, loss_night = 0.0, loss = 0.0;
		for (int i = 0; i < vectors.size(); i++)
		{
			loss_day += abs(vectors[i] - free_vectors_night[i]);
			loss_night += abs(vectors[i] - free_vectors_day[i]);
		}

		loss = min(loss_day, loss_night);

		auto predict_label = loss > 40;

		putText(tmp, to_string(loss), Point(3, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 0, 0, 0));
		putText(tmp, to_string(i), Point(3, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(255, 0, 0, 0));

#elif NON_NN_METHOD == M_COMBO
		Mat tmp, tmp2, tmp4, tmp3 = Mat::zeros(plot->rows, plot->cols, CV_8U);

		///
		///	Canny
		///

		// Blur noise
		blur(*plot, tmp, Size(5, 5));

		// Canny edge detection
		//Canny(tmp, tmp, 110, 256); //150, 255
		AutoCanny(tmp, tmp2);

		// Blur noise
		blur(tmp2, tmp, Size(4, 4));

		//threshold(tmp, tmp, 50, 255, THRESH_BINARY);

		auto positive_count = countNonZero(tmp);
		auto predict_label1 = positive_count > (size / 14); // 20

		///
		///	LBP
		///

		// Blur noise
		blur(*plot, tmp, Size(5, 5));
		AutoTreshold(tmp, tmp, 0.19, THRESH_BINARY);

		// Normalize using Local Binary Patterns
		LBP(tmp, &tmp3, min(plot->rows, plot->cols) / 8, 16);

		// Blur noise
		blur(tmp3, tmp, Size(15, 15));

		positive_count = countNonZero(tmp);
		auto black_count = size - positive_count;
		double black_white_ratio = (double)positive_count / (double)size;
		auto predict_label2 = black_white_ratio > 0.4; // 20

		///
		///	HOG
		///

		//auto plotSize = Size(plot->rows, plot->cols); //80x80
		auto block = Size(32, 32);
		auto cell = Size(16, 16);

		// Blur noise
		blur(*plot, tmp, Size(5, 5));
		LBP(tmp, &tmp2, min(plot->rows, plot->cols) / 8, 16);

		blur(*free_plot_day, tmp3, Size(5, 5));
		LBP(tmp3, &tmp4, min(free_plot_day->rows, free_plot_night->cols) / 8, 16);
		auto free_vectors_day = HOG(tmp4, &tmp3, block, cell, 9, false);

		blur(*free_plot_night, tmp3, Size(5, 5));
		LBP(tmp3, &tmp4, min(free_plot_night->rows, free_plot_night->cols) / 8, 16);
		auto free_vectors_night = HOG(tmp4, &tmp3, block, cell, 9, false);


		auto vectors = HOG(tmp2, &tmp, block, cell, 9, false);
		float loss_day = 0.0, loss_night = 0.0, loss = 0.0;
		for (int i = 0; i < vectors.size(); i++)
		{
			loss_day += abs(vectors[i] - free_vectors_night[i]);
			loss_night += abs(vectors[i] - free_vectors_day[i]);
		}

		loss = min(loss_day, loss_night);

		auto predict_label3 = loss > 40;

		auto predict_label = predict_label1 + predict_label2 * 0.9 + predict_label3 * 0.8 > 1.0;

		putText(tmp, to_string(predict_label1) + "/" + to_string(predict_label2) + "/" + to_string(predict_label3) + "=" + to_string(predict_label), Point(3, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(255, 0, 0, 0));
		putText(tmp, to_string(i), Point(3, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(255, 0, 0, 0));

#endif

		// Display grid magic
		resize(tmp, tmp2, Size(plot->cols, plot->rows));
		vconcat(*plot, tmp2, *local_spaces_img);
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


	std::vector<Mat> free_images_day;
	std::vector<Mat> free_images_night;
#if NON_NN_METHOD == M_HOG || NON_NN_METHOD == M_COMBO
	fstream free_file("free_images.txt");
	while (free_file >> test_path)
	{
		Mat free_frame = imread(test_path, 1);
		std::vector<Mat> load_images;
		cvtColor(free_frame, free_frame, COLOR_BGR2GRAY);
		extract_space(spaces, free_frame, load_images);
		if (free_images_day.size() == 0)
		{
			free_images_day = load_images;
			free_images_night = load_images;
			for (int i = 0; i < free_images_night.size(); i++) {
				free_images_night[i] /= 4;
			}
			//break;
		}
		else
		{
			for (int i = 0; i < free_images_day.size(); i++) {
				addWeighted(free_images_day[i], 0.5, load_images[i], 0.5, 0.0, free_images_day[i]);
				addWeighted(free_images_night[i], 0.5, load_images[i] / 4, 0.5, 0.0, free_images_night[i]);
			}
		}
	}
#endif

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
#if NON_NN_METHOD == M_HOG || NON_NN_METHOD == M_COMBO
			predict_label = is_occupied(i, &test_images[i], &spaces_imgs, &local_spaces_img, &spaces_img, &free_images_day[i], &free_images_night[i]);
#else
			predict_label = is_occupied(i, &test_images[i], &spaces_imgs, &local_spaces_img, &spaces_img, NULL, NULL);
#endif

			out_label_file << predict_label << endl;
			spaces[i].occup = predict_label;
		}

		// Display grid magic, missing record
		if (test_images.size() % 15 != 0)
		{
			Mat full = cv::Mat::zeros(spaces_imgs[0].size(), CV_8UC1);
			Rect roi(0, 0, local_spaces_img.cols, local_spaces_img.rows);
			local_spaces_img.copyTo(full(roi));
			spaces_imgs.push_back(full);
		}

		// Display grid magic, row merge
		const auto space_images = spaces_imgs.size() - 1;
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

bool is_in_range(Mat& src, int r, int c)
{
	return r >= 0 && r < src.rows&& c >= 0 && c < src.cols;
}

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
void AutoCanny(Mat& src, Mat& edges, float sigma)
{
	//compute the median of the single channel pixel intensities
	const auto v = Median(&src, 256);

	//apply automatic Canny edge detection using the computed median
	const auto lower = int(MAX(0, (1.0 - sigma) * v));
	const auto upper = int(MIN(255, (1.0 + sigma) * v));
	//printf("%d, %d\n", lower, upper);
	Canny(src, edges, lower, upper);
}

void AutoTreshold(Mat& src, Mat& dst, float sigma, int method)
{
	//compute the median of the single channel pixel intensities
	const auto v = Median(&src, 256);

	const auto lower = int(MAX(0, (1.0 - sigma) * v));
	const auto upper = int(MIN(255, (1.0 + sigma) * v));
	//printf("%d, %d [%d]\n", lower, upper, v);
	threshold(src, dst, lower, upper, method);
}

std::vector<float> HOG(Mat& src, Mat* dst, Size block, Size cell, int nBins, bool dominant)
{
	resize(src, *dst, src.size());

	HOGDescriptor descriptor(src.size(), block, cell, cell, nBins);
	std::vector<float> descriptors;
	descriptor.compute(src, descriptors);

	/* Ref: Fast Calculation of Histogram of Oriented Gradient Feature
	 *      by Removing Redundancy in Overlapping Block
	 *      https://github.com/zhouzq-thu/HOGImage
	 */
	 // count in the window
	int numCellsX = descriptor.winSize.width / descriptor.cellSize.width;
	int numCellsY = descriptor.winSize.height / descriptor.cellSize.height;
	int numBlocksX = (descriptor.winSize.width - descriptor.blockSize.width + descriptor.blockStride.width) / descriptor.blockStride.width;
	int numBlocksY = (descriptor.winSize.height - descriptor.blockSize.height + descriptor.blockStride.height) / descriptor.blockStride.height;

	// count in the block
	int numCellsInBlockX = descriptor.blockSize.width / descriptor.cellSize.width;
	int numCellsInBlockY = descriptor.blockSize.height / descriptor.cellSize.height;

	int sizeGrads[] = { numCellsY, numCellsX, descriptor.nbins };
	Mat gradStrengths(3, sizeGrads, CV_32F, Scalar(0));
	Mat cellUpdateCounter(numCellsY, numCellsX, CV_32S, Scalar(0));

	float* desPtr = &descriptors[0];
	for (int bx = 0; bx < numBlocksX; bx++) {
		for (int by = 0; by < numBlocksY; by++) {
			for (int cx = 0; cx < numCellsInBlockX; cx++) {
				for (int cy = 0; cy < numCellsInBlockY; cy++) {
					int cellX = bx + cx;
					int cellY = by + cy;
					int* cntPtr = &cellUpdateCounter.at<int>(cellY, cellX);
					float* gradPtr = &gradStrengths.at<float>(cellY, cellX, 0);
					(*cntPtr)++;
					for (int bin = 0; bin < descriptor.nbins; bin++) {
						float* ptr = gradPtr + bin;
						*ptr = (*ptr * (*cntPtr - 1) + *(desPtr++)) / (*cntPtr);
					}
				}
			}
		}
	}

	const float radRangePerBin = M_PI / descriptor.nbins;
	const float maxVecLen = min(descriptor.cellSize.width, descriptor.cellSize.height) / 2 * 2;


	// Darken
	*dst = *dst / 4;
	std::vector<float> bests;

	for (int cellX = 0; cellX < numCellsX; cellX++) {
		for (int cellY = 0; cellY < numCellsY; cellY++) {
			Point2f ptTopLeft = Point2f(cellX * descriptor.cellSize.width, cellY * descriptor.cellSize.height);
			Point2f ptCenter = ptTopLeft + Point2f(descriptor.cellSize) / 2;
			Point2f ptBottomRight = ptTopLeft + Point2f(descriptor.cellSize);

			// Draw rectangle grid
			/*rectangle(*dst,
				ptTopLeft,
				ptBottomRight,
				CV_RGB(100, 100, 100),
				1);*/

				// Best bin
			auto best = 0;
			auto power = gradStrengths.at<float>(cellY, cellX, 0);
			for (int bin = 1; bin < descriptor.nbins; bin++) {
				float gradStrength = gradStrengths.at<float>(cellY, cellX, bin);
				// no line to draw?
				if (gradStrength == 0)
					continue;

				if (gradStrength > power)
				{
					power = gradStrength;
					best = bin;
				}
			}

			bests.push_back(power);

			// draw the perpendicular line of the gradient
			float angle = best * radRangePerBin + radRangePerBin / 2;
			float scale = power * maxVecLen;
			Point2f direction = Point2f(sin(angle), -cos(angle));
			cv::line(*dst,
				(ptCenter - direction * scale),
				(ptCenter + direction * scale),
				CV_RGB(50, 50, 255),
				1);

			// All bins
			/*for (int bin = 0; bin < descriptor.nbins; bin++) {
				float gradStrength = gradStrengths.at<float>(cellY, cellX, bin);
				// no line to draw?
				if (gradStrength == 0)
					continue;

				// draw the perpendicular line of the gradient
				float angle = bin * radRangePerBin + radRangePerBin / 2;
				float scale = gradStrength * maxVecLen;
				Point2f direction = Point2f(sin(angle), -cos(angle));
				line(*dst,
					(ptCenter - direction * scale),
					(ptCenter + direction * scale),
					CV_RGB(50, 50, 255),
					1);
			}*/
		}
	}

	if (dominant)
		return bests;
	return descriptors;
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
		if (i > 9)
			putText(frame, to_string(i), Point(sx - 12, sy - 18), FONT_HERSHEY_COMPLEX_SMALL, 0.9, Scalar(255, 0, 0, 255), 2);
		else
			putText(frame, to_string(i), Point(sx - 5, sy - 18), FONT_HERSHEY_COMPLEX_SMALL, 0.9, Scalar(255, 0, 0, 255), 2);
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
	cout << "\nFalse" << endl;
	cout << " Positives " << falsePositives << endl;
	cout << " Negatives " << falseNegatives << endl << endl;
	cout << "True" << endl;
	cout << " Positives " << truePositives << endl;
	cout << " Negatives " << trueNegatives << endl << endl;

	const float acc = static_cast<float>(truePositives + trueNegatives) / static_cast<float>(truePositives + trueNegatives + falsePositives + falseNegatives);
	cout << "Accuracy " << acc << endl;

	const float precision = static_cast<float>(truePositives) / static_cast<float>(truePositives + falsePositives);
	const float sensitivity = static_cast<float>(truePositives) / static_cast<float>(truePositives + falseNegatives);
	const float f1 = 2 * precision * sensitivity / (precision + sensitivity);
	cout << "F1 Score " << f1 << endl << endl;
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
