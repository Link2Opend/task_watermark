#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <string>
#include <io.h>
#include <time.h>
#include "watermark.h"
#include "core.h"

using namespace std;
using namespace cv;

vector<int> wm = { 1,0,1,0,0,0,0,1,
					1,1,0,0,1,1,0,1,
					0,1,1,0,0,0,0,1};

vector<int> wm_bch = { 1,0,1,0,0,0,0,1,
					1,1,0,0,1,1,0,1,
					0,1,1,0,0,0,0,1,
					0,0,0,1,0,1,0,0,
					1,1,1,1,1,1,0,1,
					1,0,0,1,0,1,1,0,
					0,1,0,1,0,0,0,0,
					0,0,1,0,1,0,1,0,
					0,0,0,1,0,1,0,0,
					0,1,0,1,1,0,0,1,
					1,1,1,1,1,0,0,0,
					0,0,0,0,0,0,0,0 };
void getFiles(const std::string& path, std::vector<std::string>& files)
{
	//文件句柄  
	long long hFile = 0;
	//文件信息，_finddata_t需要io.h头文件  
	struct _finddata_t fileinfo;
	std::string p;
	int i = 0;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				//if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					//getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void test_embed(int quality)
{

	vector<string> files;
	getFiles("D:\\sublime text 3\\watermark\\img\\ori", files);
	WaterMark bwm = WaterMark(1, 1);
	
	bwm.read_wm(wm);
	clock_t sclock, eclock;
	int totaltime = 0;
	for (int i = 0; i < files.size(); i++)
	{
		string filename = files[i];
		sclock = clock();
		bwm.read_img(filename);
		Mat embed_img = bwm.embed();
		eclock = clock();
		totaltime += (int)(eclock - sclock);
		
		string savename = "img\\jpeg_";
		savename.append(to_string(quality)).append("\\").append(to_string(i)).append(".jpg");
		imwrite(savename, embed_img, { IMWRITE_JPEG_QUALITY, quality });
	}
	printf("%d\n", totaltime/4);
}

void test_extract(int quality)
{

	vector<string> files;
	string pathname = "img\\jpeg_";
	pathname.append(to_string(quality));
	getFiles(pathname, files);
	WaterMark bwm = WaterMark(1, 1);
	int len = wm_bch.size();
	vector<int> a;
	for (int i = 0; i < files.size(); i++)
	{
		string filename = files[i];
		cout << filename << endl;
		Mat embed_img = imread(filename);
		vector<int> wm_extract = bwm.extract(embed_img, len);
		int cou = 96;
		for (int j = 0; j < len; j++)
		{
			if (wm_bch[j] != wm_extract[j]) cou--;
		}
		cout << cou << endl;
		a.push_back(cou);
	}

	int all = 0;
	int count = 0;
	for (int i = 0; i < a.size(); i++)
	{
		all+=a[i];
		if (a[i] >= 90) count++;
	}
	cout << all/64 << endl;
	cout << count << endl;
}

void test_rotate(int angle)
{
	vector<string> files;
	getFiles("D:\\sublime text 3\\watermark\\img\\ori", files);
	WaterMark bwm = WaterMark(1, 1);

	bwm.read_wm(wm);
	for (int i = 0; i < files.size(); i++)
	{
		string filename = files[i];
		bwm.read_img(filename);
		Mat embed_img = bwm.embed();
		Mat m = getRotationMatrix2D(Point2f(embed_img.cols / 2, embed_img.rows / 2), angle, 1);
		Mat out_img;
		warpAffine(embed_img, out_img, m, Size(embed_img.cols, embed_img.rows));
		string savename = "img\\rotate_";
		savename.append(to_string(angle)).append("\\").append(to_string(i)).append(".jpg");
		imwrite(savename, out_img);
	}
}

void test_irotate(int angle)
{
	vector<string> files;
	string pathname = "img\\rotate_";
	pathname.append(to_string(angle));
	getFiles(pathname, files);
	WaterMark bwm = WaterMark(1, 1);
	int len = wm.size();
	vector<int> a;
	for (int i = 0; i < files.size(); i++)
	{
		string filename = files[i];
		cout << filename << endl;
		Mat rotate_img = imread(filename);
		Mat m = getRotationMatrix2D(Point2f(rotate_img.cols / 2, rotate_img.rows / 2), -angle, 1);
		Mat out_img;
		warpAffine(rotate_img, out_img, m, Size(rotate_img.cols, rotate_img.rows));
		/*imshow("qqq", out_img);
		waitKey(0);*/
		vector<int> wm_extract = bwm.extract(out_img, len);
		int cou = 100;
		for (int j = 0; j < len; j++)
		{
			if (wm[j] != wm_extract[j]) cou--;
		}
		cout << cou << endl;
		a.push_back(cou);
	}

	int all = 0;
	int total = 0;
	for (int i = 0; i < a.size(); i++)
	{
		if (a[i] >= 90) all++;
		total += a[i];
	}
	cout << all << endl;
	cout << total/64 << endl;
}

void test_resize(float ratio)
{
	vector<string> files;
	getFiles("D:\\sublime text 3\\watermark\\img\\ori", files);
	WaterMark bwm = WaterMark(1, 1);

	bwm.read_wm(wm);
	for (int i = 0; i < files.size(); i++)
	{
		string filename = files[i];
		bwm.read_img(filename);
		Mat embed_img = bwm.embed();
		Mat out_img;
		resize(embed_img, out_img, Size(embed_img.cols * ratio, embed_img.rows * ratio));
		
		string savename = "img\\resize_";
		savename.append(to_string((int)(ratio*10))).append("\\").append(to_string(i)).append(".jpg");
		imwrite(savename, out_img);
	}
}

void test_iresize(float ratio)
{
	vector<string> files;
	string pathname = "img\\resize_";
	pathname.append(to_string((int)(ratio*10)));
	getFiles(pathname, files);
	WaterMark bwm = WaterMark(1, 1);
	int len = wm.size();
	vector<int> a;
	ratio = 1 / ratio;
	for (int i = 0; i < files.size(); i++)
	{
		string filename = files[i];
		cout << filename << endl;
		Mat resize_img = imread(filename);
		Mat out_img;
		resize(resize_img, out_img, Size(resize_img.cols * ratio, resize_img.rows * ratio));
		/*imshow("qqq", out_img);
		waitKey(0);*/
		vector<int> wm_extract = bwm.extract(out_img, len);
		int cou = 100;
		for (int j = 0; j < len; j++)
		{
			if (wm[j] != wm_extract[j]) cou--;
		}
		cout << cou << endl;
		a.push_back(cou);
	}

	int all = 0;
	int total = 0;
	for (int i = 0; i < a.size(); i++)
	{
		if (a[i] >= 90) all++;
		total += a[i];
	}
	cout << all << endl;
	cout << total / 64 << endl;
}
int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	/*WaterMark bwm = WaterMark(1, 1);
	bwm.read_img("D:\\sublime text 3\\watermark\\img\\tmp\\1.jpg");
	vector<int> wm = { 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1 };
	clock_t start, end;
	start = clock();
	
	vector<int> bch_bit = bwm.read_wm(wm);
	for (int i = 0; i < bch_bit.size(); i++)
		cout << bch_bit[i];
	Mat embed_img = bwm.embed();
	end = clock();
	int totaltime = end - start;
	cout << totaltime << endl;
	imwrite("embed.jpg", embed_img);
	cout << "1 over" << endl;

	int len = bwm.wm_bit.size();
	WaterMark bwm1 = WaterMark(1, 1);
	Mat embed = imread("embed.jpg");
	vector<int> wm_extract = bwm1.extract(embed, len);
	for (int i = 0; i < wm_extract.size(); i++)
	{
		cout << wm_extract[i];
	}*/
	//test_embed(20);
	test_extract(80);
	//test_rotate(90);
	//test_irotate(90);
	//test_resize(1.5);
	//test_iresize(1.5);
	return 0;
}