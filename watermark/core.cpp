#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>
#include <string>

using namespace std;
using namespace cv;

class WaterMarkCore {
public:
    vector<int> block_shape = { 4, 4 };
    int password_img;
    int d1 = 36;
    int d2 = 20; //d1 / d2 越大鲁棒性越强, 但输出图片的失真越大

    //init data
    Mat img;
    Mat img_YUV;//self.img 是原图，self.img_YUV 对像素做了加白偶数化

    vector<vector<double>> ca;
    vector<vector<double>> hvd;
    vector<vector<double>> ca_block;
    vector<vector<double>> ca_part;
    self.ca, self.hvd, =[np.array([])] * 3, [np.array([])] * 3;// 每个通道 dct 的结果
    self.ca_block = [np.array([])] * 3; //每个 channel 存一个四维 array，代表四维分块后的结果
    self.ca_part = [np.array([])] * 3; //四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

    int wm_size = 0;
    int block_num = 0;//# 水印的长度，原图片可插入信息的个数

    self.pool = AutoPool(mode = mode, processes = processes);
    bool fast_mode = false;
    vector<int> alpha;//用于处理透明图
    WaterMarkCore(int password_img, string mode, string processes);

    void init_block_index();
    void read_img_arr(Mat img);
    void read_wm(vector<int> wm_bit);
    void block_add_wm(int arg);
    void block_add_wm_slow(int arg);
    void block_add_wm_fast(int arg);
    Mat embed();
    string block_get_wm(int arg);
    string block_get_wm_slow(int arg);
    string block_get_wm_fast(int arg);
    void extract_raw(Mat img);
    vector<int> extract_avg(vector<int> wm_block_bit);
    Mat extract(Mat img, vector<int> wm_shape);
    Mat extract_with_kmeans(Mat img, vector<int> wm_shape);
};

