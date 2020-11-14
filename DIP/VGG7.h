#pragma once
// https://blog.paperspace.com/popular-deep-learning-architectures-alexnet-vgg-googlenet/
// https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c

#include "Method.h"
#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

using VGG7_XS = dlib::loss_multiclass_log<
	dlib::relu < dlib::fc < 2,
	dlib::dropout <
	dlib::relu < dlib::fc < 1024,
	dlib::dropout <
	dlib::relu < dlib::fc < 1024,
	dlib::max_pool < 3, 3, 1, 1,
	dlib::relu < dlib::con < 256, 3, 3, 1, 1,
	dlib::relu < dlib::con < 256, 3, 3, 1, 1,
	dlib::max_pool < 3, 3, 1, 1,
	dlib::relu < dlib::con < 128, 3, 3, 1, 1,
	dlib::max_pool < 3, 3, 1, 1,
	dlib::relu < dlib::con < 64, 3, 3, 1, 1,
	dlib::input < dlib::matrix < dlib::rgb_pixel
	>>>>>>>>>>>>>>>>>>>>>>; // 224x224x3


class VGG7 : public Method
{
public:
	static VGG7_XS net;
	static const std::string save;
	static bool hasNet;

	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override
	{
		std::vector<dlib::matrix<dlib::rgb_pixel>> images;
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
			dlib::cv_image<dlib::rgb_pixel> cimg(image);
			dlib::matrix<dlib::rgb_pixel> dlibimg = dlib::mat(cimg);
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

			//int len;
			//net.print(std::cout, 0, len);
		}
		cv::Mat image;
		prepare(input, image);
		dlib::cv_image<dlib::rgb_pixel> cimg(image);
		dlib::matrix<dlib::rgb_pixel> dlibimg = dlib::mat(cimg);

		auto prediction = net(dlibimg);

		input.copyTo(output);
		putText(output, std::to_string(prediction), cv::Point(3, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 0, 0, 0));
		putText(output, std::to_string(i), cv::Point(3, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0, 0));

		return prediction;
	};

private:
	static void prepare(cv::Mat& input, cv::Mat& result)
	{
		cv::Mat tmp;
		cv::cvtColor(input, tmp, cv::COLOR_GRAY2RGB);
		//cv::blur(tmp, tmp, cv::Size(5, 5));
		//result.convertTo(tmp, CV_32FC3);
		cv::resize(tmp, result, cv::Size(32, 32));
		//cv::resize(input, result, cv::Size(113, 113));
	}

	static dlib::dnn_trainer<VGG7_XS>* prepareTrainer()
	{
		dlib::dnn_trainer<VGG7_XS>* trainer = new dlib::dnn_trainer<VGG7_XS>(net, dlib::sgd(), { 0 });
		trainer->set_learning_rate(1e-5);
		trainer->set_min_learning_rate(1e-7);
		trainer->set_mini_batch_size(32);
		trainer->set_iterations_without_progress_threshold(1000);
		trainer->set_max_num_epochs(200);

		return trainer;
	}
};

VGG7_XS VGG7::net = VGG7_XS{};
const std::string VGG7::save = "vgg7_4.dat";
bool VGG7::hasNet = false;