#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>
#include <string>

using namespace std;
using namespace cv;


class tools
{
public:
	tools();
	~tools();
	vector<vector<int>> random_strategy1(int seed, int size, int block_shape);
	vector<int> one_dim_kmeans(vector<double>& inputs);
	vector<Mat> haar_dwt(Mat& matrix);
	Mat haar_idwt(Mat& ca, vector<Mat>& hvd);
	vector<vector<Mat>> as_strided(Mat& matrix, vector<int>& shape);
	vector<int> RandomArray(vector<int>& oldArray, int password);
};
