#include <iostream>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <string>
#include "watermark.h"
#include "core.h"

using namespace std;
using namespace cv;


int main()
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
	WaterMark bwm = WaterMark(1, 1);
	vector<int> wm = { 1,0,1,0,0,0,0,1,1,1,0,0,1,1 };
	bwm.read_img("D:\\sublime text 3\\watermark\\img\\tmp\\1.jpg");
	bwm.read_wm(wm);
	Mat embed_img = bwm.partition_embed();
	imwrite("embed.jpg", embed_img, {IMWRITE_JPEG_QUALITY,100});
	cout << "1 over" << endl;
	int len = bwm.wm_bit.size();
	vector<vector<int>> wm_extract = bwm.partition_extract(embed_img, len);
	for (int i = 0; i < wm_extract.size(); i++)
	{
		for (int j = 0; j < wm_extract[i].size(); j++)
		{
			cout << wm_extract[i][j];
		}
		cout << endl;
	}

	return 0;
}