#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>
#include <string>
using namespace std;
using namespace cv;

class WaterMark {
public:
	int blind_wm = 0;
	string password_wm;
	int wm_size = 0;
	int wm_bit = 0;
	WaterMark(int password_wm = 1, int password_img = 1, vector<int> block_shape = {4,4}, string mode = "common")
	{
		cout << "hello?" << endl;
	}

	Mat read_img(string filename, Mat img)
	{

	}
};


