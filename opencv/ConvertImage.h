#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;

class ConvertImage
{
public:
	static void mat2array(Mat img, uchar *data); // Mat�^����z�񂾂����o��]
	static void edge(uchar *data, int width, int hight, uchar *output); // �G�b�W���o
};