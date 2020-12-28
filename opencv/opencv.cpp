#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <fstream>
#include "ConvertImage.h"
#include "extractionEdge.h"
using namespace std;
using namespace cv;

// ログバッファー
stringstream logBuf;

// 関数宣言
void writeLog();

int main(int argc, char* argv[])
{
	// ファイルのパスを取得
	string readPath;
	string outputPath;

#ifdef _DEBUG
	// デバックモード
	readPath = "C:\\Users\\rinna\\Documents\\Cpp\\Projects\\Edge\\opencv\\photos\\testImage2.jpg";
	outputPath = "C:\\Users\\rinna\\Documents\\Cpp\\Projects\\Edge\\x64\\Debug\\photos\\result_cp.jpg";
#else
	// リリースモード
	// もしなんも入力されてなかったら終わる
	if (argv[1] == NULL || argv[2] == NULL)
	{
		logBuf << "ERROR: NoPath or NoPath Output Folder";
		logBuf << "INFO : Exit the program" << endl;
		return -1;
	}

	readPath = argv[1];
	outputPath = (string)argv[2] + "result.jpg";
#endif

	// 計測
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	LARGE_INTEGER start, end;

	// イメージの読み込み
	Mat img;
	img = imread(readPath);

	// ファイルがあるかの確認
	if (img.data == NULL)
	{
		logBuf << "ERROR: ImageFile not opend" << endl;
		logBuf << "INFO : Exit the program" << endl;
		writeLog();
		return -1;
	}

	// イメージを配列化
	int hight = img.rows;
	int width = img.cols;
	int pixels = width * hight;

	unsigned char *data = new unsigned char[pixels * 3];
	ConvertImage::mat2array(img, data);

	// コンフィグファイルの読み込み
	FILE *cfg;
	errno_t err = fopen_s(&cfg, "config.cfg", "r");

	// 開けなかったらログを出す
	if (err != 0)
	{
		logBuf << "ERROR: ConfigFile not opend" << endl;
		logBuf << "INFO : Use the initial value" << endl;
	}
	else
	{
		extractionEdge::configRead(cfg);
	}

	// 計測開始
	QueryPerformanceCounter(&start);

	// 画像加工
	unsigned char *edgeData = new unsigned char[pixels];
	extractionEdge::edge(data, width, hight, edgeData);
	delete[] data;

	// 計測終了
	QueryPerformanceCounter(&end);

	// 加工済み配列をMat型に変換
	Mat convertedImg(hight, width, CV_8U, edgeData);

	double time = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	logBuf << "INFO : time_" << time << "ms" << endl;

	// イメージの保存
	imwrite(outputPath, convertedImg);

	delete[] edgeData;

	// ログファイルの保存
	logBuf << "INFO : SUCCSESS!" << endl;
	writeLog();

	return 0;
}

// ログファイルの保存
void writeLog()
{
	// ファイルを開く
	FILE *log;
	fopen_s(&log, "log.log", "w");
	// 書き込み
	fputs(logBuf.str().c_str(), log);
	fclose(log);
}