#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>
#include <string>
#include "core.h"
#include "tools.h"
#include <cmath>

using namespace std;
using namespace cv;


WaterMarkCore::WaterMarkCore(int password_img)
{
    this->password_img = password_img;
    d1 = 50;
    d2 = 20; //d1 / d2 越大鲁棒性越强, 但输出图片的失真越大
    block_shape = { 4,4 };
    wm_size = 0;
    block_num = 0;
}

WaterMarkCore::WaterMarkCore()
{
    this->password_img = 1;
    d1 = 50;
    d2 = 20; //d1 / d2 越大鲁棒性越强, 但输出图片的失真越大
    block_shape = { 4,4 };
    wm_size = 0;
    block_num = 0;
}

WaterMarkCore::~WaterMarkCore()
{
}

void WaterMarkCore::init_block_index()
{
    block_num = ca_block_shape[0] * ca_block_shape[1];
    assert(wm_size < block_num, "水印信息超过可嵌入数据量\n");
    part_shape = { ca_block_shape[0] * block_shape[0],ca_block_shape[1] * block_shape[1] };
    for (int i = 0; i < ca_block_shape[0]; i++)
    {
        for (int j = 0; j < ca_block_shape[1]; j++)
        {
            vector<int> tmp = { i,j };
            block_index.emplace_back(tmp);
        }
    }
}


void WaterMarkCore::read_img_arr(Mat& img)
{
    block_index.clear();
    ca.clear();
    hvd.clear();
    ca_block.clear();

    img.convertTo(this->img, CV_32FC3);
    img_shape = { img.rows,img.cols };
    Mat tmp;
    cvtColor(this->img, tmp, COLOR_BGR2YUV);
    copyMakeBorder(tmp, this->img_YUV, 0, this->img.rows % 2, 0, this->img.cols % 2, BORDER_CONSTANT, Scalar(0, 0, 0));
    ca_shape = { (img.rows + 1) / 2,(img.cols + 1) / 2 };
    ca_block_shape = { ca_shape[0] / block_shape[0], ca_shape[1] / block_shape[1], block_shape[0], block_shape[1] };
    vector<Mat> channels;
    split(this->img_YUV, channels);
    for (int ch = 0; ch < 3; ch++)
    {
        vector<Mat> cahvd = tool.haar_dwt(channels[ch]);
        ca.emplace_back(cahvd[0]);
        vector<Mat> hvd_tmp;
        for (int i = 0; i < 3; i++)
        {
            hvd_tmp.emplace_back(cahvd[i + 1]);
        }
        hvd.emplace_back(hvd_tmp);
        ca_block.emplace_back(tool.as_strided(cahvd[0], ca_block_shape));
    }
}


void WaterMarkCore::read_wm(vector<int>& wm_bit)
{
    this->wm_size = wm_bit.size();
    this->wm_bit = wm_bit;
}

Mat WaterMarkCore::block_add_wm(Mat& block, vector<int>& shuffer, int i)
{
    int wm_1 = wm_bit[i % wm_size];
    Mat dct_output;
    dct(block, dct_output);
    //if (i < 20) cout << wm_1 << endl;
    Mat S, U, V;
    SVD::compute(dct_output, S, U, V);
    float* ptr = S.ptr<float>(0);
    //if (i < 20) cout << S << endl;
    ptr[0] = (int(ptr[0] / (d1*1.0)) + 1.0 / 4 + 1.0 / 2 * wm_1) * d1;
    //cout << ptr[0] << endl;
    Mat idct_output;
    idct(U * S.diag(S) * V, idct_output);
    return idct_output;
}

Mat WaterMarkCore::embed()
{
    init_block_index();

    vector<Mat> embed_ca;
    for (int i = 0; i < ca.size(); i++) embed_ca.emplace_back(ca[i].clone());
    vector<Mat> embed_YUV;
    vector<vector<int>> id_shuffle = tool.random_strategy1(password_img, block_num, block_shape[0]*block_shape[1]);

    for (int channel = 0; channel < 3; channel++)
    {
        vector<Mat> tmp;
        for (int i = 0; i < block_num; i++)
        {
            Mat tmp_res = block_add_wm(ca_block[channel][block_index[i][0]][block_index[i][1]], id_shuffle[i], i);
            tmp.emplace_back(tmp_res);
        }

        for (int i = 0; i < block_num; i++)
        {
            ca_block[channel][block_index[i][0]][block_index[i][1]] = tmp[i];
        }
        vector<Mat> row_tmp;
        for (int i = 0; i < ca_block_shape[0]; i++)
        {
            Mat row_res;
            hconcat(ca_block[channel][i], row_res);
            row_tmp.emplace_back(row_res);
        }

        Mat ch_res;
        vconcat(row_tmp, ch_res);
        for (int i = 0; i < part_shape[0]; i++)
        {
            float* embed_ca_row_ptr = embed_ca[channel].ptr<float>(i);
            float* ch_res_row_ptr = ch_res.ptr<float>(i);
            for (int j = 0; j < part_shape[1]; j++)
            {
                embed_ca_row_ptr[j] = ch_res_row_ptr[j];
            }
        }
        embed_YUV.emplace_back(tool.haar_idwt(embed_ca[channel], hvd[channel]));
    }
    Mat embed_img_YUV;
    merge(embed_YUV, embed_img_YUV);
    embed_img_YUV = embed_img_YUV(Rect(0, 0, img_shape[1], img_shape[0]));
    Mat embed_img;
    cvtColor(embed_img_YUV, embed_img, COLOR_YUV2BGR);
    embed_img.convertTo(embed_img, CV_8UC3);

    return embed_img;
}

int WaterMarkCore::block_get_wm(Mat& block, vector<int> shuffler, int i)
{
    Mat dct_output;
    dct(block, dct_output);
    Mat S, U, V;
    SVD::compute(dct_output, S, U, V);
    float* ptr = S.ptr<float>(0);
    float a = fmod(ptr[0],d1);
    float b = d1 / 2.0;
    int wm = a > b ? 1 : 0;
    //cout << wm << endl;
    return wm;
}

vector<vector<int>> WaterMarkCore::extract_raw(Mat& img)
{
    read_img_arr(img);
    init_block_index();

    vector<vector<int>> wm_block_bit(3, vector<int>(this->block_num));
    vector<vector<int>> id_shuffle = tool.random_strategy1(password_img, this->block_num, block_shape[0] * block_shape[1]);
    for (int channel = 0; channel < 3; channel++)
    {
        for (int i = 0; i < this->block_num; i++)
        {
            wm_block_bit[channel][i] = block_get_wm(ca_block[channel][block_index[i][0]][block_index[i][1]], id_shuffle[i], i);
        }
    }
    return wm_block_bit;
}

vector<double> WaterMarkCore::extract_avg(vector<vector<int>> wm_block_bit)
{
    vector<double> wm_avg(this->wm_size, 0);
    for (int i = 0; i < this->wm_size; i++)
    {
        double sum = 0.0;
        double count = 0.0;
        for (int j = 0; j < wm_block_bit.size(); j++)
        {
            int cou = 0;
            for (int k = i; k < wm_block_bit[j].size(); k++)
            {
                if (cou % this->wm_size == 0)
                {
                    sum += 1.0*wm_block_bit[j][k];
                    cou = 0;
                    count++;
                }
                cou++;
            }
        }
        wm_avg[i] = sum / (count*1.0);
        //cout << wm_avg[i]<<" "<<sum<<" "<<count << endl;
    }
    return wm_avg;
}

vector<double> WaterMarkCore::extract(Mat img, int wm_shape)
{
    wm_size = wm_shape;
    vector<vector<int>> wm_block_bit = extract_raw(img);
    vector<double> wm_avg = extract_avg(wm_block_bit);
    return wm_avg;
}

vector<int> WaterMarkCore::extract_with_kmeans(Mat img, int wm_shape)
{
    vector<double> wm_avg = extract(img,wm_shape);
    return tool.one_dim_kmeans(wm_avg);
}
