#include "ConvertImage.h"

// 画像データーを配列に変換
void ConvertImage::mat2array(Mat img, uchar *data)
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

// 画像データーからエッジを検出
void ConvertImage::edge(uchar *data, int width, int hight, uchar *output)
{
	int pixels = width * hight;

	// エッジ検出
	float *side = new float[pixels];
	float *vertical = new float[pixels];
	#pragma omp parallel for
	for (int y = 0; y < hight; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int target	= (x + y * width) * 3;
			int left	= target - 3;
			int right	= target + 3;
			int up		= target - width * 3;
			int down	= target + width * 3;

			int sideMin = (width * 3) * y;
			int sideMax = (width * 3) * (y + 1);
			int downMax = (width * 3 * hight) + x * 3;

			// 真ん中
			int tr = data[target + 0];
			int tg = data[target + 1];
			int tb = data[target + 2];

			// 左
			int lr = (left < sideMin) ? data[right + 0] : data[left + 0];
			int lg = (left < sideMin) ? data[right + 1] : data[left + 1];
			int lb = (left < sideMin) ? data[right + 2] : data[left + 2];

			// 左側との差の割合
			float averageLr = abs(lr - tr) / 255.0f;
			float averageLg = abs(lg - tg) / 255.0f;
			float averageLb = abs(lb - tb) / 255.0f;
			// 一番差が大きいものをとる
			float averageLeft =	 (averageLr < averageLg) ?
								 (averageLg < averageLb) ? averageLb : averageLg 
								:(averageLr < averageLb) ? averageLb : averageLr;

			// 右
			int rr = (sideMax <= right) ? data[left + 0] : data[right + 0];
			int rg = (sideMax <= right) ? data[left + 1] : data[right + 1];
			int rb = (sideMax <= right) ? data[left + 2] : data[right + 2];

			// 右側との差の割合
			float averageRr = abs(rr - tr) / 255.0f;
			float averageRg = abs(rg - tg) / 255.0f;
			float averageRb = abs(rb - tb) / 255.0f;
			// 一番差が大きいものをとる
			float averageRight =	 (averageRr < averageRg) ? 
									 (averageRg < averageRb) ? averageRb : averageRg 
									:(averageRr < averageRb) ? averageRb : averageRr;

			// 左右面との差の平均
			float differenceS = (averageLeft + averageRight) / 2.0f;

			side[x + y * width] = differenceS;

			// 上
			int ur = (up < 0) ? data[down + 0] : data[up + 0];
			int ug = (up < 0) ? data[down + 1] : data[up + 1];
			int ub = (up < 0) ? data[down + 2] : data[up + 2];

			// 上側との差の割合
			float averageUr = abs(ur - tr) / 255.0f;
			float averageUg = abs(ug - tg) / 255.0f;
			float averageUb = abs(ub - tb) / 255.0f;
			// 一番差が大きいものをとる
			float averageUp =	 (averageUr < averageUg) ? 
								 (averageUg < averageUb) ? averageUb : averageUg
								:(averageUr < averageUb) ? averageUb : averageUr;

			// 下
			int dr = (downMax <= down) ? data[up + 0] : data[down + 0];
			int dg = (downMax <= down) ? data[up + 1] : data[down + 1];
			int db = (downMax <= down) ? data[up + 2] : data[down + 2];

			// 下側との差の割合
			float averageDr = abs(dr - tr) / 255.0f;
			float averageDg = abs(dg - tg) / 255.0f;
			float averageDb = abs(db - tb) / 255.0f;
			// 一番差が大きいものをとる
			float averageDown =	 (averageDr < averageDg) ? 
								 (averageDg < averageDb) ? averageDb : averageDg
								:(averageDr < averageDb) ? averageDb : averageDr;
			
			// 上下面との差の平均
			float differenceV = (averageUp + averageDown) / 2.0f;

			vertical[x + y * width] = differenceV;
		}
	}

	// 合成
	#pragma omp parallel for
	for (int i = 0; i < pixels; i++)
	{
		float difference = side[i] + vertical[i];
		float d = (0.18f < difference) ? 0.0f : 1.0f;
		output[i] = (uchar)(d * 255);
	}
	delete[] side;
	delete[] vertical;
}