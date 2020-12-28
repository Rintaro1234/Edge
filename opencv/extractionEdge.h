#pragma once
#include <opencv2/opencv.hpp>

class extractionEdge
{
private:
	static float edgePercentage;

public:
	static void edge(unsigned char *data, int width, int hight, unsigned char *output); // エッジ抽出
	static void configRead(FILE *cfg);
};