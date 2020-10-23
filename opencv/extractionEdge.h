#pragma once
#include <opencv2/opencv.hpp>

class extractionEdge
{
public:
	static void edge(uchar *data, int width, int hight, uchar *output); // エッジ抽出
};

