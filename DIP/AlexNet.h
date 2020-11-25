#pragma once
#include "Method.h"
#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

using alex = dlib::loss_multiclass_log<
	dlib::relu < dlib::fc < 2,
	dlib::dropout <
	dlib::relu < dlib::fc < 4096,
	dlib::dropout <
	dlib::relu < dlib::fc < 4096,
	dlib::max_pool < 3, 3, 2, 2,
	dlib::relu < dlib::con < 384, 3, 3, 1, 1,
	dlib::relu < dlib::con < 384, 3, 3, 1, 1,
	dlib::relu < dlib::con < 384, 3, 3, 1, 1,
	dlib::max_pool < 3, 3, 2, 2,
	dlib::relu < dlib::con < 256, 5, 5, 1, 1,
	dlib::max_pool < 3, 3, 2, 2,
	dlib::relu < dlib::con < 96, 11, 11, 4, 4,
	dlib::input < dlib::matrix < dlib::rgb_pixel
	>>>>>>>>>>>>>>>>>>>>>>>>; // 227x227x3


class AlexNet : public Method
{
public:
	static alex net;
	static const std::string save;
	static bool hasNet;
	
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override
	{
		std::vector<dlib::matrix<dlib::rgb_pixel>>* images = new std::vector<dlib::matrix<dlib::rgb_pixel>>();
		std::vector<unsigned long>* labels = new std::vector<unsigned long>();

		//augment_data(train_images, train_labels);

		for (int i = 0; i < train_images.size(); i++) {
			cv::Mat image;
			prepare(train_images[i], image);
			dlib::cv_image<dlib::rgb_pixel> cimg(image);
			dlib::matrix<dlib::rgb_pixel> dlibimg = dlib::mat(cimg);
			images->push_back(dlibimg);
			labels->push_back(train_labels[i]);
		}

		auto trainer = prepareTrainer();
		trainer->be_verbose();

		std::cout << "Training ..." << std::endl;

		trainer->train(*images, *labels);
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

	void augment_data(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels)
	{
		std::cout << "Data Augmentation ..." << std::endl;

		// -- Add Night images
		const auto images_count_0 = train_images.size();
		for (int i = 0; i < images_count_0; i++)
		{
			train_images.push_back(train_images[i] / 4);
			train_labels.push_back(train_labels[i]);
		}

		// -- Add Flipped images
		const auto images_count = train_images.size();
		for (int i = 0; i < images_count; i++)
		{
			cv::Mat original = train_images[i];
			cv::Mat changed; // 0 - X, 1+ - Y
			cv::flip(original, changed, 0); // X
			train_images.push_back(changed);
			train_labels.push_back(train_labels[i]);
			cv::flip(changed, changed, 1); // XY
			train_images.push_back(changed);
			train_labels.push_back(train_labels[i]);
			cv::flip(original, changed, 1); // Y
			train_images.push_back(changed);
			train_labels.push_back(train_labels[i]);
		}

		std::cout << "Train images: " << train_images.size() << std::endl;
		std::cout << "Train labels: " << train_labels.size() << std::endl;
	}

private:
	static void prepare(cv::Mat& input, cv::Mat& result)
	{
		cv::Mat tmp;
		cv::cvtColor(input, tmp, cv::COLOR_GRAY2RGB);
		//result.convertTo(tmp, CV_32FC3);
		cv::resize(tmp, result, cv::Size(227, 227));
		//cv::resize(input, result, cv::Size(113, 113));
	}

	static dlib::dnn_trainer<alex>* prepareTrainer()
	{
		dlib::dnn_trainer<alex>* trainer = new dlib::dnn_trainer<alex>(net, dlib::sgd(), { 0 });
		trainer->set_learning_rate(1e-5);
		trainer->set_min_learning_rate(1e-8);
		trainer->set_mini_batch_size(256);
		trainer->set_iterations_without_progress_threshold(1000);
		trainer->set_max_num_epochs(200);

		return trainer;
	}
};

alex AlexNet::net = alex{};
const std::string AlexNet::save = "alex3.dat";
bool AlexNet::hasNet = false;