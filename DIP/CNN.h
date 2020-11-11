#pragma once
#include "Method.h"
#include "dlib/matrix.h"
#include "dlib/dnn.h"
#include "dlib/opencv.h"


// I 80x80x3 = 19200, C 80x80x1 = 6400, H 80x80x1 = 6400, All 80x80x5 = 32000
using cnn_xl = dlib::loss_multiclass_log<
	dlib::relu < dlib::fc < 2,
	dlib::relu < dlib::fc < 4,
	dlib::relu < dlib::fc < 8,
	dlib::relu < dlib::fc < 16,
	dlib::relu < dlib::fc < 32,
	dlib::relu < dlib::fc < 64,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 128, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 128, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 256, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 256, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 512, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 512, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 1024, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 1024, 5, 5, 1, 1,
	dlib::input <dlib::matrix < unsigned char
	>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

using cnn_xs = dlib::loss_multiclass_log<
	dlib::fc<2,
	dlib::relu< dlib::fc<84,
	dlib::relu< dlib::fc<120,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con < 16, 5, 5, 1, 1,
	dlib::max_pool < 2, 2, 2, 2, dlib::relu < dlib::con< 6, 5, 5, 1, 1,
	dlib::input< dlib::matrix<unsigned char>>
	>>>>>>>>>>>>;

cnn_xs net{};
bool hasNet = false;

class CNN : public Method
{
public:
	bool NeedTraining() override { return true; }
	
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override
	{
		std::vector<dlib::matrix<unsigned char>> images;
		std::vector<unsigned long> labels;

		for (int i = 0; i < train_images.size(); i++) {
			dlib::cv_image<unsigned char> cimg(train_images[i]);
			dlib::matrix<unsigned char> dlibimg = dlib::mat(cimg);
			images.push_back(dlibimg);
			labels.push_back(train_labels[i]);
		}

		dlib::dnn_trainer<cnn_xs> trainer(net, dlib::sgd(), { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
		//dnn_trainer<ichl> trainer(net);
		trainer.set_learning_rate(0.01);
		trainer.set_min_learning_rate(0.0001);
		trainer.set_mini_batch_size(512);
		trainer.set_iterations_without_progress_threshold(2000);
		trainer.set_max_num_epochs(5);
		trainer.be_verbose();

		std::cout << "Training ..." << std::endl;

		trainer.train(images, labels);
		hasNet = true;

		std::cout << "Saving ..." << std::endl;

		dlib::serialize("ichl.dat") << net;
	};

	bool predict(int i, cv::Mat& input, cv::Mat& output) override
	{
		if (!hasNet) {
			dlib::deserialize("ichl.dat") >> net;
			hasNet = true;
			
			int len;
			net.print(std::cout, 0, len);
		}

		dlib::cv_image<unsigned char> cimg(input);
		dlib::matrix<unsigned char> dlibimg = dlib::mat(cimg);
		
		output = input;

		auto prediction = net(dlibimg);
		auto prediction2 = *net.subnet()(dlibimg).host();
		std::cout << prediction << " " << prediction2 << ", ";
		return prediction2 > 0.5;
	};
};
