#pragma once
#include <opencv2/ml.hpp>

#include "Method.h"

cv::Ptr<cv::ml::SVM> svm;
bool svmSet = false;
cv::HOGDescriptor svmcHog(cv::Size(80, 80), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
const int cmax = 320;
const float eps = 1e-7;

class SVMC : public Method
{
public:	
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override
	{
		PrepareSVM();

		cv::Mat sampleFeatureMat;
		cv::Mat sampleLabelMat;
		int DescriptorDim;

		DescriptorDim = train_images.size();
		for (int i = 0; i < DescriptorDim; i++)
		{
			train_images.push_back(train_images[i] / 4);
			train_labels.push_back(train_labels[i]);
		}

		for(int i = 0; i < train_images.size(); i++)
		{
			std::vector<float> descriptors;
			svmcHog.compute(preprocess(train_images[i]), descriptors, cv::Size(8, 8));

			if (0 == i)
			{
				//Dimension of HOG descriptor
				DescriptorDim = descriptors.size();
				sampleFeatureMat = cv::Mat::zeros(train_images.size(), DescriptorDim, CV_32FC1);
				sampleLabelMat = cv::Mat::zeros(train_images.size(), 1, CV_32SC1);
			}
			//Copy the calculated HOG descriptor to the sample feature matrix sampleFeatureMat  
			for (int i = 0; i < DescriptorDim; i++)
			{
				sampleFeatureMat.at<float>(i, i) = descriptors[i];
			}
			sampleLabelMat.at<float>(i, 0) = train_labels[i];
		}
		
		cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(sampleFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);
		svm->train(td);
		
		svm->save("svm.data");
		svmSet = true;
	};
	
	bool predict(int i, cv::Mat& input, cv::Mat& output) override
	{
		if(!svmSet)
		{
			PrepareSVM();
			svm->load("svm.data");
			svmSet = true;
		}

		output = preprocess(input);

		std::vector<float> descriptors;
		svmcHog.compute(output, descriptors);
		cv::Mat testDescriptor = cv::Mat::zeros(1, descriptors.size(), CV_32FC1);
		for (size_t i = 0; i < descriptors.size(); i++)
		{
			testDescriptor.at<float>(0, i) = descriptors[i];
		}
		auto predict = svm->predict(testDescriptor);

		putText(output, std::to_string(predict), cv::Point(3, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 0, 0, 0));
		putText(output, std::to_string(i), cv::Point(3, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0, 0));
		
		std::cout << predict << ", ";
		return predict > 0.5;
	};

private:
	static void PrepareSVM()
	{

		svm = cv::ml::SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::SIGMOID);
		svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, cmax, eps));

		/*svm->setCoef0(0.0);
		svm->setDegree(3);
		svm->setGamma(0);
		svm->setNu(0.5);
		svm->setP(0.9);
		svm->setC(1.0);*/
	}

	
	static cv::Mat preprocess(cv::Mat img)
	{
		cv::Mat tmp, tmp2 = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		cv::blur(img, tmp, cv::Size(5, 5));

		Threshold::LBP(tmp, &tmp2, std::min(img.rows, img.cols) / 8, 16);
		//Threshold::LBP(tmp, &tmp2, 4, 16);

		return tmp2;
	}
};
