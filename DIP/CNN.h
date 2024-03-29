﻿#pragma once
#include "Method.h"
#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

using cnn_xs = dlib::loss_multiclass_log<
	dlib::fc<2,
	dlib::relu< dlib::fc<84,
	dlib::relu< dlib::fc<120,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 16, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con< 6, 5, 5, 1, 1,
	dlib::input< dlib::matrix<unsigned char>>
	>>>>>>>>>>>>; // 28x28x1

class CNN : public Method
{	
public:
	static cnn_xs net;
	static const std::string save;
	static bool hasNet;
	
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override
	{
		std::vector<dlib::matrix<unsigned char>> images;
		std::vector<unsigned long> labels;

		/*const auto images_count = train_images.size();
		for (int i = 0; i < images_count; i++)
		{
			train_images.push_back(train_images[i] / 2);
			train_labels.push_back(train_labels[i]);
			train_images.push_back(train_images[i] / 3);
			train_labels.push_back(train_labels[i]);
			train_images.push_back(train_images[i] / 4);
			train_labels.push_back(train_labels[i]);
		}*/

		for (int i = 0; i < train_images.size(); i++) {
			cv::Mat image;
			prepare(train_images[i], image);
			dlib::cv_image<unsigned char> cimg(image);
			dlib::matrix<unsigned char> dlibimg = dlib::mat(cimg);
			images.push_back(dlibimg);
			labels.push_back(train_labels[i]);
		}

		auto trainer = prepareTrainer();
		trainer->be_verbose();

		std::cout << "Training ..." << std::endl;

		trainer->train(images, labels);
		hasNet = true;

		std::cout << "Saving ..." << std::endl;

		dlib::serialize(save) << net;

		delete trainer;
	};

	bool predict(int i, cv::Mat& input, cv::Mat& output) override
	{
		if (!hasNet) {
			dlib::deserialize(save) >> net;
			hasNet = true;
			
			int len;
			net.print(std::cout, 0, len);
		}
		cv::Mat image;
		prepare(input, image);
		dlib::cv_image<unsigned char> cimg(image);
		dlib::matrix<unsigned char> dlibimg = dlib::mat(cimg);

		auto prediction = net(dlibimg);

		input.copyTo(output);
		putText(output, std::to_string(prediction), cv::Point(3, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 0, 0, 0));
		putText(output, std::to_string(i), cv::Point(3, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0, 0));

		return prediction;
	};

private:
	static void prepare(cv::Mat& input, cv::Mat& result)
	{
		cv::resize(input, result, cv::Size(28, 28));
	}

	static dlib::dnn_trainer<cnn_xs>* prepareTrainer()
	{
		dlib::dnn_trainer<cnn_xs>* trainer = new dlib::dnn_trainer<cnn_xs>(net, dlib::sgd(), { 0 });
		trainer->set_learning_rate(0.01);
		trainer->set_min_learning_rate(0.0001);
		trainer->set_mini_batch_size(512);
		trainer->set_iterations_without_progress_threshold(2000);
		trainer->set_max_num_epochs(100);

		return trainer;
	}
};


cnn_xs CNN::net = cnn_xs{};
const std::string CNN::save = "cnn4.dat";
bool CNN::hasNet = false;
