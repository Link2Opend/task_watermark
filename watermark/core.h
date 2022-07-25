#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>
#include <string>
#include "tools.h"

using namespace std;
using namespace cv;

class WaterMarkCore {
public:
    vector<int> block_shape;
    int password_img;
    int d1;
    int d2; //d1 / d2 越大鲁棒性越强, 但输出图片的失真越大

    //init data
    Mat img;
    Mat img_YUV;//self.img 是原图，self.img_YUV 对像素做了加白偶数化
    tools tool;

    vector<int> img_shape;
    vector<Mat> ca;
    vector<vector<Mat>> hvd;
    vector<vector<vector<Mat>>> ca_block;
    vector<Mat> ca_part;
    vector<int> part_shape;
    vector<int> ca_shape;
    vector<int> ca_block_shape;
    vector<vector<int>> block_index;
    //self.ca, self.hvd, =[np.array([])] * 3, [np.array([])] * 3;// 每个通道 dct 的结果
    //self.ca_block = [np.array([])] * 3; //每个 channel 存一个四维 array，代表四维分块后的结果
    //self.ca_part = [np.array([])] * 3; //四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

    int wm_size;
    vector<int> wm_bit;
    int block_num;//# 水印的长度，原图片可插入信息的个数

    //self.pool = AutoPool(mode = mode, processes = processes);
    bool fast_mode = false;
    vector<int> alpha;//用于处理透明图
    WaterMarkCore(int password_img);
    WaterMarkCore();
    ~WaterMarkCore();

    void init_block_index();
    void read_img_arr(Mat& img);
    void read_wm(vector<int>& wm_bit);
    Mat block_add_wm(Mat& block, vector<int>& shuffer, int i);
    Mat embed();
    int block_get_wm(Mat& block, vector<int> shuffler, int i);
    vector<vector<int>> extract_raw(Mat& img);
    vector<double> extract_avg(vector<vector<int>> wm_block_bit);
    vector<double> extract(Mat img, int wm_shape);
    vector<int> extract_with_kmeans(Mat img, int wm_shape);
};
