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
    int d2; //d1 / d2 Խ��³����Խǿ, �����ͼƬ��ʧ��Խ��

    //init data
    Mat img;
    Mat img_YUV;//self.img ��ԭͼ��self.img_YUV ���������˼Ӱ�ż����
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
    //self.ca, self.hvd, =[np.array([])] * 3, [np.array([])] * 3;// ÿ��ͨ�� dct �Ľ��
    //self.ca_block = [np.array([])] * 3; //ÿ�� channel ��һ����ά array��������ά�ֿ��Ľ��
    //self.ca_part = [np.array([])] * 3; //��ά�ֿ����ʱ����������һ���֣�self.ca_part ������һ���ֵ� self.ca

    int wm_size;
    vector<int> wm_bit;
    int block_num;//# ˮӡ�ĳ��ȣ�ԭͼƬ�ɲ�����Ϣ�ĸ���

    //self.pool = AutoPool(mode = mode, processes = processes);
    bool fast_mode = false;
    vector<int> alpha;//���ڴ���͸��ͼ
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
