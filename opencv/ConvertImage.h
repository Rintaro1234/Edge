#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

class ConvertImage
{
public:
	static void mat2array(Mat img, unsigned char *data); // Mat型から配列だけ取り出す]
};