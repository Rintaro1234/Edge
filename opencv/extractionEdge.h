#pragma once
#include <opencv2/opencv.hpp>

class extractionEdge
{
private:
	static float edgePercentage;

public:
	static void edge(uchar *data, int width, int hight, uchar *output); // エッジ抽出
	static void configRead(FILE *cfg);
};