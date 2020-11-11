#pragma once
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>

class Method
{
public:
	virtual bool NeedTraining() { return false; }
	virtual bool CustomTraining() { return false; }
	
	virtual ~Method() = default;
	
	virtual void train(std::vector<cv::Mat>& train_images, std::vector<unsigned char>& train_labels) = 0;
	virtual bool predict(int i, cv::Mat& input, cv::Mat& output) = 0;
};


struct space
{
	int x01, y01, x02, y02, x03, y03, x04, y04, occup;
};

class Utils
{
public:
	static const int spaces_num = 56;

	void static extract_space(space* spaces, cv::Mat in_mat, std::vector<cv::Mat>& vector_images)
	{
		cv::Size space_size(80, 80);
		for (int x = 0; x < spaces_num; x++)
		{
			cv::Mat src_mat(4, 2, CV_32F);
			cv::Mat out_mat(space_size, CV_8U, 1);
			src_mat.at<float>(0, 0) = spaces[x].x01;
			src_mat.at<float>(0, 1) = spaces[x].y01;
			src_mat.at<float>(1, 0) = spaces[x].x02;
			src_mat.at<float>(1, 1) = spaces[x].y02;
			src_mat.at<float>(2, 0) = spaces[x].x03;
			src_mat.at<float>(2, 1) = spaces[x].y03;
			src_mat.at<float>(3, 0) = spaces[x].x04;
			src_mat.at<float>(3, 1) = spaces[x].y04;

			cv::Mat dest_mat(4, 2, CV_32F);
			dest_mat.at<float>(0, 0) = 0;
			dest_mat.at<float>(0, 1) = 0;
			dest_mat.at<float>(1, 0) = out_mat.cols;
			dest_mat.at<float>(1, 1) = 0;
			dest_mat.at<float>(2, 0) = out_mat.cols;
			dest_mat.at<float>(2, 1) = out_mat.rows;
			dest_mat.at<float>(3, 0) = 0;
			dest_mat.at<float>(3, 1) = out_mat.rows;

			cv::Mat H = findHomography(src_mat, dest_mat, 0);
			warpPerspective(in_mat, out_mat, H, space_size);

			/*imshow("in_mat", in_mat);
			imshow("out_mat", out_mat);
			cv::waitKey(100);*/

			vector_images.push_back(out_mat);
		}
	}

	int static load_parking_geometry(const char* filename, space* spaces)
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


	void static draw_detection(space* spaces, cv::Mat& frame)
	{
		int sx, sy;
		for (int i = 0; i < spaces_num; i++)
		{
			cv::Point pt1, pt2;
			pt1.x = spaces[i].x01;
			pt1.y = spaces[i].y01;
			pt2.x = spaces[i].x03;
			pt2.y = spaces[i].y03;
			sx = (pt1.x + pt2.x) / 2;
			sy = (pt1.y + pt2.y) / 2;
			if (spaces[i].occup)
			{
				circle(frame, cv::Point(sx, sy - 25), 12, CV_RGB(255, 0, 0), -1);
			}
			else
			{
				circle(frame, cv::Point(sx, sy - 25), 12, CV_RGB(0, 255, 0), -1);
			}
			if (i > 9)
				putText(frame, std::to_string(i), cv::Point(sx - 12, sy - 18), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, cv::Scalar(255, 0, 0, 255), 2);
			else
				putText(frame, std::to_string(i), cv::Point(sx - 5, sy - 18), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, cv::Scalar(255, 0, 0, 255), 2);
		}
	}

	static void convert_images_to_ml(const std::vector<cv::Mat>& train_samples, cv::Mat& trainData)
	{
		//--Convert data
		const int rows = (int)train_samples.size();
		const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
		cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
		trainData = cv::Mat(rows, cols, CV_32FC1);
		std::vector<cv::Mat >::const_iterator itr = train_samples.begin();
		std::vector<cv::Mat >::const_iterator end = train_samples.end();
		for (int i = 0; itr != end; ++itr, ++i)
		{
			CV_Assert(itr->cols == 1 || itr->rows == 1);
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

	static void convert_labels_to_ml(const std::vector<unsigned char>& train_labels, cv::Mat& trainLabels)
	{
		//--Convert data
		const int rows = (int)train_labels.size();
		const int cols = 1;
		cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
		trainLabels = cv::Mat(rows, cols, CV_32FC1);
		std::vector<unsigned char>::const_iterator itr = train_labels.begin();
		std::vector<unsigned char>::const_iterator end = train_labels.end();
		for (int i = 0; itr != end; ++itr, ++i)
		{
			trainLabels.at<float>(i, 0) = train_labels[i];
		}
	}
};
