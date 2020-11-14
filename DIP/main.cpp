#define OMP_NUM_THREADS 12
#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>

//opencv - https://opencv.org/
#include <opencv2/opencv.hpp>

//dlib - http://dlib.net
#include "config.h"
#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

#include "Canny.h"
#include "Threshold.h"
#include "HOG.h"
#include "SVM.h"
#include "CNN.h"
#include "AlexNet.h"
#include "VGG7.h"

using namespace std;
using namespace cv;
//using namespace cv::ml;
using namespace dlib;

void evaluation(fstream& detectorOutputFile, fstream& groundTruthFile);

void train_parking();
void test_parking();

void convert_to_ml(const std::vector< cv::Mat >& train_samples, cv::Mat& trainData);

// -- Methods --
#define M_COMBO -1
#define M_COMBO2 -2

#define M_CANNY 0
#define M_TLBP 1

#define M_HOG 10

#define M_SVM 20
#define M_CNN 21
#define M_ALEX 22
#define M_VGG7 23
// -- Methods --

#define METHOD M_VGG7

#define TRAIN true
#define WAIT 500


int main(int argc, char** argv)
{
	cout << "Train OpenCV Start" << endl;
	train_parking();
	cout << "Train OpenCV End" << endl;

	cout << "Test OpenCV Start" << endl;
	test_parking();
	cout << "Test OpenCV End" << endl;
}



void train_parking()
{
	std::vector<Mat> train_images;
	std::vector<unsigned char> train_labels;

	std::vector<Method*> ms;
	#if METHOD == M_HOG:
	ms.push_back(new HOGD{});
	#elif METHOD == M_CNN:
	ms.push_back(new CNN{});
	#elif METHOD == M_ALEX:
	ms.push_back(new AlexNet{});
	#elif METHOD == M_VGG7:
	ms.push_back(new VGG7{});
	#elif METHOD == M_SVM:
	ms.push_back(new SVMC{});
	#elif METHOD == M_COMBO:
	//m = new SVMC{};
	ms.push_back(new HOGD{});
	#elif METHOD == M_COMBO2:
	ms.push_back(new HOGD{});
	ms.push_back(new AlexNet{});
	#endif

	auto stop = true;
	for(auto m : ms)
	{
		if (m == nullptr)
			continue;

		const auto train = TRAIN || m->NeedTraining();
		
		if (train && m->CustomTraining())
			m->train(train_images, train_labels);
		else if(train)
			stop = false;
	}

	if(stop)
		return;

	//load parking lot geometry
	space* spaces = new space[Utils::spaces_num];
	Utils::load_parking_geometry("parking_map.txt", spaces);

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
		Utils::extract_space(spaces, frame, train_images);

		//training label for each parking space
		for (int i = 0; i < Utils::spaces_num; i++)
		{
			train_labels.push_back(label);
		}

	}

	delete spaces;

	cout << "Train images: " << train_images.size() << endl;
	cout << "Train labels: " << train_labels.size() << endl;

	for (auto m : ms)
	{
		if(m == nullptr)
			continue;

		const auto train = TRAIN || m->NeedTraining();
		
		if (train && !m->CustomTraining())
			m->train(train_images, train_labels);
	}

	/*delete m;
	m = nullptr;*/
}

struct PredictionMethod
{
	Method* method;
	float weight;
};

bool is_occupied(int i, Mat* plot, std::vector<Mat>* spaces_imgs, Mat* spaces_img, Mat* local_spaces_img)
{
	cv::Mat tmp = cv::Mat::zeros(plot->rows, plot->cols, CV_8UC1), tmp2;
	float min_weight = 1.0;
	std::vector<PredictionMethod> ms;
	#if METHOD == M_CANNY
		ms.push_back(PredictionMethod{ new CannyED{}, 1 });
	#elif METHOD == M_TLBP
		ms.push_back(PredictionMethod{ new Threshold{}, 1 });
	#elif METHOD == M_HOG
		ms.push_back(PredictionMethod{ new HOGD{}, 1 });
	#elif METHOD == M_SVM
		ms.push_back(PredictionMethod{ new SVMC{}, 1 });
	#elif METHOD == M_CNN
		ms.push_back(PredictionMethod{ new CNN{}, 1 });
	#elif METHOD == M_ALEX
		ms.push_back(PredictionMethod{ new AlexNet{}, 1 });
	#elif METHOD == M_VGG7
		ms.push_back(PredictionMethod{ new VGG7{}, 1 });
	#elif METHOD == M_COMBO
		ms.push_back(PredictionMethod{ new CannyED{}, 1 });
		ms.push_back(PredictionMethod{ new Threshold{}, 0.9 });
		ms.push_back(PredictionMethod{ new HOGD{}, 0.8 });
	#elif METHOD == M_COMBO2
		ms.push_back(PredictionMethod{ new CannyED{}, 1 });
		ms.push_back(PredictionMethod{ new Threshold{}, 0.9 });
		ms.push_back(PredictionMethod{ new AlexNet{}, 0.9 });
		ms.push_back(PredictionMethod{ new HOGD{}, 0.8 });
		min_weight = 2.0;
	#endif

	float weight = 0.0;

	for (auto pm : ms)
		if (pm.method != nullptr)
			weight += pm.method->predict(i, *plot, tmp)*pm.weight;


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

	return weight >= min_weight;
}

/**
 * \brief
 */
void test_parking()
{
	space* spaces = new space[Utils::spaces_num];
	Utils::load_parking_geometry("parking_map.txt", spaces);

	fstream test_file("test_images.txt");
	ofstream out_label_file("out_prediction.txt");
	string test_path;

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
		Utils::extract_space(spaces, frame, test_images);

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
		Utils::draw_detection(spaces, draw_frame);
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
