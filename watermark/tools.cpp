#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>
#include <string>
#include "tools.h"

using namespace std;
using namespace cv;

struct MyTuple
{
    int x;
    int y;
    MyTuple(int a, int b) :x(a), y(b) {};
};

tools::tools()
{
}
tools::~tools()
{
}

vector<vector<int>> tools::random_strategy1(int seed, int size, int block_shape)
{
    srand(seed);
    vector<vector<int>> res;
    for (int i = 0; i < size; i++)
    {
        vector<MyTuple> tmp;
        for (int j = 0; j < block_shape; j++)
        {
            MyTuple tmp_num = MyTuple(rand(), j);
            tmp.push_back(tmp_num);
        }
        sort(tmp.begin(), tmp.end(), [](MyTuple a, MyTuple b) { return a.x < b.x; });
        vector<int> tmp_res;
        for (int j = 0; j < tmp.size(); j++)
        {
            tmp_res.push_back(tmp[j].y);
        }
        res.push_back(tmp_res);
    }
    return res;
}


vector<int> tools::one_dim_kmeans(vector<double>& inputs)
{
    double threshold = 0.0;
    double e_tol = 1e-6;
    vector<double> center = {*min_element(inputs.begin(), inputs.end()), *max_element(inputs.begin(), inputs.end())};
    vector<int> res(inputs.size());
    for (int i = 0; i < 300; i++)
    {
        threshold = (center[0] + center[1]) / 2.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        int cou = 0;
        for (int j = 0; j < inputs.size(); j++)
        {
            if (inputs[j] > threshold)
            {
                sum2 += inputs[j];
                cou++;
            }
            else
            {
                sum1 += inputs[j];
            }
        }
        center = { sum1 / ((inputs.size() - cou)*1.0), sum2 / (cou*1.0) };

        if (abs((center[0] + center[1]) / 2 - threshold) < e_tol)
        {
            threshold = (center[0] + center[1]) / 2;
            break;
        }
    }
    for (int i = 0; i < inputs.size(); i++)
    {
        if (inputs[i] > threshold) res[i] = 1;
        else res[i] = 0;
    }
    return res;
}


vector<Mat> tools::haar_dwt(Mat& matrix)
{
    int rows = matrix.rows;
    int cols = matrix.cols;
    int rowsize = rows / 2;
    int colsize = cols / 2;
    matrix.convertTo(matrix, CV_32FC1);
    Mat A = Mat(rowsize, colsize, CV_32FC1);
    Mat B = Mat(rowsize, colsize, CV_32FC1);
    Mat C = Mat(rowsize, colsize, CV_32FC1);
    Mat D = Mat(rowsize, colsize, CV_32FC1);
    for (int i = 0; i < rowsize; i++)
    {
        float* row1 = matrix.ptr<float>(i * 2);
        float* row2 = matrix.ptr<float>(i * 2 + 1);
        float* Arow = A.ptr<float>(i);
        float* Brow = B.ptr<float>(i);
        float* Crow = C.ptr<float>(i);
        float* Drow = D.ptr<float>(i);
        for (int j = 0; j < colsize; j++)
        {
            double a = row1[j * 2];
            double b = row1[j * 2 + 1];
            double c = row2[j * 2];
            double d = row2[j * 2 + 1];
            Arow[j] = 0.5 * (a + b + c + d);
            Brow[j] = 0.5 * (a + b - c - d);
            Crow[j] = 0.5 * (a - b + c - d);
            Drow[j] = 0.5 * (a - b - c + d);
        }
    }
    vector<Mat> res = { A,B,C,D };
    return res;
}

Mat tools::haar_idwt(Mat& ca, vector<Mat>& hvd)
{
    int rowsize = ca.rows * 2;
    int colsize = ca.cols * 2;
    Mat res(ca.rows * 2, ca.cols * 2, CV_32FC1);
    for (int i = 0; i < ca.rows; i++)
    {
        float* row1 = res.ptr<float>(i * 2);
        float* row2 = res.ptr<float>(i * 2 + 1);
        float* Arow = ca.ptr<float>(i);
        float* Brow = hvd[0].ptr<float>(i);
        float* Crow = hvd[1].ptr<float>(i);
        float* Drow = hvd[2].ptr<float>(i);
        for (int j = 0; j < ca.cols; j++)
        {
            double a = Arow[j];
            double b = Brow[j];
            double c = Crow[j];
            double d = Drow[j];
            row1[j * 2] = 0.5 * (a + b + c + d);
            row1[j * 2 + 1] = 0.5 * (a + b - c - d);
            row2[j * 2] = 0.5 * (a - b + c - d);
            row2[j * 2 + 1] = 0.5 * (a - b - c + d);
        }
    }
    return res;
}

vector<vector<Mat>> tools::as_strided(Mat& matrix, vector<int>& shape)
{
    vector<vector<Mat>> res(shape[0], vector<Mat>(shape[1]));
    for (int i = 0; i < shape[0]; i++)
    {
        for (int j = 0; j < shape[1]; j++)
        {
            res[i][j] = matrix(Rect(j * shape[3], i * shape[2], shape[3], shape[2]));
        }
    }
    return res;
}

vector<int> tools::RandomArray(vector<int>& oldArray, int password)
{
    vector<int> tmp = oldArray;
    srand(password);
    vector<int> res;
    for (int i = oldArray.size(); i > 0; i--)
    {
        int index = rand() % i;
        res.push_back(tmp[index]);
        tmp.erase(tmp.begin() + index);
    }
    return res;
}
