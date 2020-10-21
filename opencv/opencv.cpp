﻿#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include "ConvertImage.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	// もしなんも入力されてなかったら終わる
	/*if (argv[1] == NULL)
	{
		cout << "NoPath";
		return 1;
	}

	cout << argv[1] << endl;*/

	// 計測
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	LARGE_INTEGER start, end;

	// イメージの読み込み
	Mat img;
	//string path = argv[1];
	img = imread("photos/testImage3.jpg");

	// イメージを配列化
	int hight = img.rows;
	int width = img.cols;
	int pixels = width * hight;

	uchar *data = new uchar[pixels * 3];
	ConvertImage::mat2array(img, data);

	// 計測開始
	QueryPerformanceCounter(&start);

	// 画像加工
	uchar *edgeData = new uchar[pixels];
	ConvertImage::edge(data, width, hight, edgeData);
	delete[] data;

	// 計測終了
	QueryPerformanceCounter(&end);

	// 加工済み配列をMat型に変換
	Mat c(hight, width, CV_8U, edgeData);

	double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	cout << time << "ms" << endl;

	// イメージの表示
	imshow("c", c);
	waitKey(0);
	imwrite("photos/c.jpg", c);

	delete[] edgeData;

	return 0;
}