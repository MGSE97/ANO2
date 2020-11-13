#pragma once
#include "Method.h"
#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"

#define NET 1

// I 80x80x3 = 19200, C 80x80x1 = 6400, H 80x80x1 = 6400, All 80x80x5 = 32000
#if NET == 0

#define INPUT_TYPE unsigned char
using cnn_xs = dlib::loss_multiclass_log<
	dlib::fc<2,
	dlib::relu< dlib::fc<84,
	dlib::relu< dlib::fc<120,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 16, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con< 6, 5, 5, 1, 1,
	dlib::input< dlib::matrix<INPUT_TYPE>>
	>>>>>>>>>>>>; // 28x28x1

cnn_xs net{};
const std::string save = "cnn4.dat";
#endif

#if NET == 1

#define INPUT_TYPE dlib::rgb_pixel
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
	dlib::input < dlib::matrix < INPUT_TYPE
	>>>>>>>>>>>>>>>>>>>>>>>>;

alex net{};
const std::string save = "alex4.dat";
#endif

bool hasNet = false;

class CNN : public Method
{
private:
	static void prepare(cv::Mat &input, cv::Mat &result)
	{
		// -------------- CNN XS ----------------
		#if NET == 0
		cv::resize(input, result, cv::Size(28, 28));
		#endif

		// -------------- ALEX NET ----------------
		#if NET == 1
		cv::Mat tmp;
		cv::cvtColor(input, tmp, cv::COLOR_GRAY2RGB);
		//result.convertTo(tmp, CV_32FC3);
		cv::resize(tmp, result, cv::Size(227, 227));
		//cv::resize(input, result, cv::Size(113, 113));
		#endif
	}


	#if NET == 0
		static dlib::dnn_trainer<cnn_xs>* prepareTrainer()
		{
			// -------------- CNN XS ----------------
			dlib::dnn_trainer<cnn_xs>* trainer = new dlib::dnn_trainer<cnn_xs>(net, dlib::sgd(), { 0 });
			trainer->set_learning_rate(0.01);
			trainer->set_min_learning_rate(0.0001);
			trainer->set_mini_batch_size(512);
			trainer->set_iterations_without_progress_threshold(2000);
			trainer->set_max_num_epochs(100);

			return trainer;
		}
	#endif

	#if NET == 1
		static dlib::dnn_trainer<alex>* prepareTrainer()
		{
			// -------------- ALEX NET ----------------
			dlib::dnn_trainer<alex>* trainer = new dlib::dnn_trainer<alex>(net, dlib::sgd(), { 0 });
			trainer->set_learning_rate(1e-4);
			trainer->set_min_learning_rate(1e-6);
			trainer->set_mini_batch_size(128);
			trainer->set_iterations_without_progress_threshold(1000);
			trainer->set_max_num_epochs(200);

			return trainer;
		}
	#endif
	
public:
	bool NeedTraining() override { return true; }
	
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override
	{
		std::vector<dlib::matrix<INPUT_TYPE>> images;
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
			dlib::cv_image<INPUT_TYPE> cimg(image);
			dlib::matrix<INPUT_TYPE> dlibimg = dlib::mat(cimg);
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
		dlib::cv_image<INPUT_TYPE> cimg(image);
		dlib::matrix<INPUT_TYPE> dlibimg = dlib::mat(cimg);

		auto prediction = net(dlibimg);


		input.copyTo(output);
		putText(output, std::to_string(prediction), cv::Point(3, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 0, 0, 0));
		putText(output, std::to_string(i), cv::Point(3, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0, 0));

		return prediction;
	};
};
