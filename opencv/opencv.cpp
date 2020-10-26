#ifdef _DEBUG
#pragma comment(lib, "C:\\opencv\\build\\x64\\vc15\\lib\\opencv_world440d.lib")
#else
#pragma comment(lib, "C:\\opencv\\build\\x64\\vc15\\lib\\opencv_world440.lib")
#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include "ConvertImage.h"
#include "extractionEdge.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	string readPath;
	string outputPath;
#ifdef _DEBUG
	readPath = "C:\\Users\\rinna\\Documents\\Cpp\\Projects\\Edge\\opencv\\photos\\testImage11.jpg";
	outputPath = "C:\\Users\\rinna\\Documents\\Cpp\\Projects\\Edge\\x64\\Debug\\photos\\result.jpg";
#else
	// もしなんも入力されてなかったら終わる
	if (argv[1] == NULL || argv[2] == NULL)
	{
		cout << "NoPath or NoPath Output Folder";
		return 1;
	}

	cout << argv[1] << endl;
	readpath = argv[1];
	outputPath = (string)argv[2] + "result.jpg";
#endif

	// 計測
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	LARGE_INTEGER start, end;

	// イメージの読み込み
	Mat img;
	img = imread(readPath);

	// イメージを配列化
	int hight = img.rows;
	int width = img.cols;
	int pixels = width * hight;

	uchar *data = new uchar[pixels * 3];
	ConvertImage::mat2array(img, data);

	// コンフィグファイルの読み込み
	FILE *cfg;
	errno_t err = fopen_s(&cfg, "config.cfg", "r");

	if (err != 0)
	{
		cout << "Error: ConfigFile not opend" << endl;
	}
	extractionEdge::configRead(cfg);

	// 計測開始
	QueryPerformanceCounter(&start);

	// 画像加工
	uchar *edgeData = new uchar[pixels];
	extractionEdge::edge(data, width, hight, edgeData);
	delete[] data;

	// 計測終了
	QueryPerformanceCounter(&end);

	// 加工済み配列をMat型に変換
	Mat convertedImg(hight, width, CV_8U, edgeData);

	double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	cout << time << "ms" << endl;

	// イメージの保存
	imwrite(outputPath, convertedImg);

	delete[] edgeData;

	return 0;
}