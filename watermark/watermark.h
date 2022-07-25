#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>
#include <string>
#include "core.h"
#include "tools.h"
using namespace std;
using namespace cv;

class WaterMark {
public:
	int blind_wm;
	int password_wm;
	int wm_size;
	Mat embedding_img;
	tools tool;
	vector<int> wm_bit;
	WaterMarkCore wm_core;

	WaterMark(int password_wm, int password_img);
	~WaterMark();
	Mat read_img(string filename);
	void read_wm(vector<int> wm_content);
	Mat embed();
	Mat partition_embed();
	vector<int> extract_decrypt(vector<int>& wm_avg);
	vector<int> extract(string filename, int wm_shape);
	vector<int> extract(Mat& embed_img, int wm_shape);
	vector<vector<int>> partition_extract(string filename, int wm_shape);
	vector<vector<int>> partition_extract(Mat& embed_img, int wm_shape);
	
};
