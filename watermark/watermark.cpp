#include "tools.h"
#include "watermark.h"
using namespace std;
using namespace cv;


WaterMark::WaterMark(int password_wm, int password_img)
{
	wm_core = WaterMarkCore(password_img);
	this->password_wm = password_wm;
	blind_wm = 0;
	wm_size = 0;
}
WaterMark::~WaterMark()
{
}
Mat WaterMark::read_img(string filename)
{
	Mat img = imread(filename,IMREAD_UNCHANGED);
	embedding_img = img.clone();
	wm_core.read_img_arr(img);
	return img;
}

void WaterMark::read_wm(vector<int> wm_content)
{
	wm_bit = wm_content;
	wm_size = wm_bit.size();
	wm_bit = tool.RandomArray(wm_bit, password_wm);
	wm_core.read_wm(wm_bit);
}

Mat WaterMark::embed()
{
	Mat embed_img = wm_core.embed();
	return embed_img;
}

Mat WaterMark::partition_embed()
{
	int rowclip = 8;
	int colclip = 8;
	int rows = embedding_img.rows / rowclip;
	int cols = embedding_img.cols / colclip;
	vector<Mat> col_img;
	for (int i = 0; i < rowclip; i++)
	{
		vector<Mat> row_img;
		for (int j = 0; j < colclip; j++)
		{
			Mat part_clip = embedding_img(Rect(j * cols, i * rows, cols, rows));
			/*imshow("0", part_clip);
			waitKey(0);*/
			wm_core.read_img_arr(part_clip);
			Mat embed_part = wm_core.embed();
			row_img.emplace_back(embed_part);
		}
		Mat row_res;
		hconcat(row_img, row_res);
		col_img.emplace_back(row_res);
	}
	Mat embed_img;
	vconcat(col_img, embed_img);
	return embed_img;
}

vector<int> WaterMark::extract_decrypt(vector<int>& wm_avg)
{
	vector<int> wm_index(wm_size);
	for (int i = 0; i < wm_index.size(); i++)
	{
		wm_index[i] = i;
	}
	wm_index = tool.RandomArray(wm_index, password_wm);
	vector<int> res(wm_size);
	for (int i = 0; i < wm_index.size(); i++)
	{
		res[wm_index[i]] = wm_avg[i];
	}
	return res;

}

vector<int> WaterMark::extract(string filename, int wm_shape)
{
	Mat embed_img = imread(filename,IMREAD_COLOR);
	vector<int> wm_avg = wm_core.extract_with_kmeans(embed_img, wm_shape);
	vector<int> wm = extract_decrypt(wm_avg);
	return wm;
}

vector<int> WaterMark::extract(Mat& embed_img, int wm_shape)
{
	vector<int> wm_avg = wm_core.extract_with_kmeans(embed_img, wm_shape);
	vector<int> wm = extract_decrypt(wm_avg);
	return wm;
}

vector<vector<int>> WaterMark::partition_extract(string filename, int wm_shape)
{
	int row = 135;
	int col = 240;
	Mat embed_img = imread(filename, IMREAD_COLOR);
	vector<vector<int>> wm_set;
	for (int i = 0; i < embed_img.rows - row; i++)
	{
		for (int j = 0; j < embed_img.cols - col; j++)
		{
			Mat part_img = embed_img(Rect(j, i, col, row));
			vector<int> wm_avg = wm_core.extract_with_kmeans(part_img, wm_shape);
			vector<int> wm = extract_decrypt(wm_avg);
			wm_set.emplace_back(wm);
		}
	}

	return wm_set;
}

vector<vector<int>> WaterMark::partition_extract(Mat& embed_img, int wm_shape)
{
	int row = 135;
	int col = 240;
	vector<vector<int>> wm_set;
	for (int i = 0; i < embed_img.rows - row; i++)
	{
		for (int j = 0; j < embed_img.cols - col; j++)
		{
			Mat part_img = embed_img(Rect(j, i, col, row));
			vector<int> wm_avg = wm_core.extract_with_kmeans(part_img, wm_shape);
			vector<int> wm = extract_decrypt(wm_avg);
			for (int l = 0; l < wm.size(); l++)
			{
				cout << wm[l];
				if (l == wm.size() - 1) cout << endl;
			}
			imshow(" 0", part_img);
			waitKey(0);
			wm_set.emplace_back(wm);
		}
	}
	
	return wm_set;
}