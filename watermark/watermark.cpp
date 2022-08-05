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
	prim_poly = 211;
	t = 10;
	m = 0;
	unsigned int tmp = prim_poly;
	while (tmp >>= 1) {
		m++;
	}
	//bch = init_bch(m, t, prim_poly);
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

vector<int> WaterMark::read_wm(vector<int> wm_content)
{
	wm_bit = wm_content;
	struct bch_control* bch = init_bch(m, t, prim_poly);
	uint8_t buf[20];
	memset(buf, 0, 20);
	for (int i = 0; i < wm_bit.size() / 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			buf[i] |= (wm_bit[i * 8 + j] << (7 - j));
		}
	}
	unsigned int a = bch->ecc_bytes;
	unsigned int len = 3;
	uint8_t code[9];
	memset(code, 0, a);
	encode_bch(bch, buf, len, code);
	vector<int> ecc(a * 8);
	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			ecc[i * 8 + (7 - j)] = code[i] & 0x1;
			code[i] >>= 1;
		}
	}
	for (int i = 0; i < ecc.size(); i++) wm_bit.emplace_back(ecc[i]);

	wm_size = wm_bit.size();
	vector<int> res = wm_bit;
	wm_bit = tool.RandomArray(wm_bit, password_wm);

	wm_core.read_wm(wm_bit);
	return res;
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
	wm_size = wm_shape;
	vector<int> wm_avg = wm_core.extract_with_kmeans(embed_img, wm_shape);
	vector<int> wm = extract_decrypt(wm_avg);
	return wm;
}

vector<int> WaterMark::extract(Mat& embed_img, int wm_shape)
{
	vector<int> wm_avg = wm_core.extract_with_kmeans(embed_img, wm_shape);
	wm_size = wm_shape;
	vector<int> wm = extract_decrypt(wm_avg);

	struct bch_control* bch = init_bch(m, t, prim_poly);
	uint8_t buf[3];
	memset(buf, 0, 3);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			buf[i] |= (wm[i * 8 + j] << (7 - j));
		}
	}

	uint8_t code[9];
	memset(code, 0, 9);
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			code[i] |= (wm[24 + i * 8 + j] << (7 - j));
		}
	}


	unsigned int a = bch->ecc_bytes;
	unsigned int len = 3;

	unsigned int* errloc;
	errloc = (unsigned int*)alloca(10 * sizeof(unsigned int));

	int count = 0;
	count = decode_bch(bch, buf, len, code, NULL, NULL, errloc);

	if (count > 0)
	{
		for (int i = 0; i < count; i++)
		{
			if (errloc[i] < 3 * 8)
			{
				buf[errloc[i] >> 3] ^= (1 << (errloc[i] & 7));		//纠错过程
			}
			else code[(errloc[i] >> 3) - 3] ^= (1 << (errloc[i] & 7));
		}

		vector<int> wm_correct(96);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				wm_correct[i * 8 + (7 - j)] = buf[i] & 0x1;
				buf[i] >>= 1;
			}
		}
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				wm_correct[24 + i * 8 + (7 - j)] = code[i] & 0x1;
				code[i] >>= 1;
			}
		}
		return wm_correct;
	}

	return wm;
}

vector<vector<int>> WaterMark::partition_extract(string filename, int wm_shape)
{
	int row = 135;
	int col = 240;
	wm_size = wm_shape;
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
	wm_size = wm_shape;
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