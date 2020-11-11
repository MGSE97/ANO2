#pragma once
#include <fstream>
#include <opencv2/opencv.hpp>

#include "Method.h"
#include "Threshold.h"

std::vector<cv::Mat> free_images_day;
std::vector<cv::Mat> free_images_night;

class HOGD : public Method
{
public:
	bool NeedTraining() override { return true; }
	bool CustomTraining() override { return true; }
	
	void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) override
	{
		const auto spaces = new space[Utils::spaces_num];
		Utils::load_parking_geometry("parking_map.txt", spaces);
		
		std::fstream free_file("free_images.txt");
		std::string test_path;
		while (free_file >> test_path)
		{
			cv::Mat free_frame = cv::imread(test_path, 0);
			std::vector<cv::Mat> load_images;
			//cvtColor(free_frame, free_frame, cv::COLOR_BGR2GRAY);
			Utils::extract_space(spaces, free_frame, load_images);
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
					cv::addWeighted(free_images_day[i], 0.5, load_images[i], 0.5, 0.0, free_images_day[i]);
					cv::addWeighted(free_images_night[i], 0.5, load_images[i] / 4, 0.5, 0.0, free_images_night[i]);
				}
			}
		}
		delete spaces;
	};
	
	bool predict(int i, cv::Mat& input, cv::Mat& output) override
	{
		cv::Mat tmp, tmp2, tmp3, tmp4;

		//auto plotSize = Size(plot->rows, plot->cols); //80x80
		auto block = cv::Size(32, 32);
		auto cell = cv::Size(16, 16);

		// Blur noise
		blur(input, tmp, cv::Size(5, 5));
		Threshold::LBP(tmp, &tmp2, std::min(input.rows, input.cols) / 8, 16);

		blur(free_images_day[i], tmp3, cv::Size(5, 5));
		Threshold::LBP(tmp3, &tmp4, std::min(free_images_day[i].rows, free_images_day[i].cols) / 8, 16);
		auto free_vectors_day = CvHOG(tmp4, &tmp3, block, cell, 9, false);

		blur(free_images_night[i], tmp3, cv::Size(5, 5));
		Threshold::LBP(tmp3, &tmp4, std::min(free_images_night[i].rows, free_images_night[i].cols) / 8, 16);
		auto free_vectors_night = CvHOG(tmp4, &tmp3, block, cell, 9, false);
		

		auto vectors = CvHOG(tmp2, &tmp, block, cell, 9, false);
		output = tmp;
		
		float loss_day = 0.0, loss_night = 0.0, loss = 0.0;
		for (int i = 0; i < vectors.size(); i++)
		{
			loss_day += abs(vectors[i] - free_vectors_night[i]);
			loss_night += abs(vectors[i] - free_vectors_day[i]);
		}

		loss = std::min(loss_day, loss_night);

		auto predict_label = loss > 40;

		putText(output, std::to_string(loss), cv::Point(3, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(255, 0, 0, 0));
		putText(output, std::to_string(i), cv::Point(3, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, cv::Scalar(255, 0, 0, 0));
		
		return predict_label;
	};

	std::vector<float> CvHOG(cv::Mat& src, cv::Mat* dst, cv::Size block, cv::Size cell, int nBins, bool dominant)
	{
		resize(src, *dst, src.size());

		cv::HOGDescriptor descriptor(src.size(), block, cell, cell, nBins);
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
		cv::Mat gradStrengths(3, sizeGrads, CV_32F, cv::Scalar(0));
		cv::Mat cellUpdateCounter(numCellsY, numCellsX, CV_32S, cv::Scalar(0));

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
		const float maxVecLen = std::min(descriptor.cellSize.width, descriptor.cellSize.height) / 2 * 2;


		// Darken
		*dst = *dst / 4;
		std::vector<float> bests;

		for (int cellX = 0; cellX < numCellsX; cellX++) {
			for (int cellY = 0; cellY < numCellsY; cellY++) {
				cv::Point2f ptTopLeft = cv::Point2f(cellX * descriptor.cellSize.width, cellY * descriptor.cellSize.height);
				cv::Point2f ptCenter = ptTopLeft + cv::Point2f(descriptor.cellSize) / 2;
				cv::Point2f ptBottomRight = ptTopLeft + cv::Point2f(descriptor.cellSize);

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
				cv::Point2f direction = cv::Point2f(sin(angle), -cos(angle));
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
};
