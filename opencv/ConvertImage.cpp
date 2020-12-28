#include "ConvertImage.h"

// 画像データーを配列に変換
void ConvertImage::mat2array(Mat img, unsigned char *data)
{
	// BGRからRGBに変換
	cvtColor(img, img, COLOR_BGR2RGB);

	int hight = img.rows;
	int width = img.cols;
	int pixels = width * hight;

	// ピクセル数分の要素を持つ配列に三次元(rgb)を入れる
	Vec3b *rgb = new Vec3b[pixels];
	#pragma omp parallel for
	for (int y = 0; y < hight; y++)
	{
		Vec3b *b = img.ptr<Vec3b>(y);
		for (int x = 0; x < width; x++)
		{
			rgb[x + y * width] = b[x];
		}
	}

	// 三次元配列を一次元配列に変換
	#pragma omp parallel for
	for (int i = 0; i < pixels; i++)
	{
		data[i * 3 + 0] = rgb[i][0];
		data[i * 3 + 1] = rgb[i][1];
		data[i * 3 + 2] = rgb[i][2];
	}
	delete[] rgb;
}