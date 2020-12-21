#include "ConvertImage.h"

// �摜�f�[�^�[��z��ɕϊ�
void ConvertImage::mat2array(Mat img, uchar *data)
{
	// BGR����RGB�ɕϊ�
	cvtColor(img, img, COLOR_BGR2RGB);

	int hight = img.rows;
	int width = img.cols;
	int pixels = width * hight;

	// �s�N�Z�������̗v�f�����z��ɎO����(rgb)������
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

	// �O�����z����ꎟ���z��ɕϊ�
	#pragma omp parallel for
	for (int i = 0; i < pixels; i++)
	{
		data[i * 3 + 0] = rgb[i][0];
		data[i * 3 + 1] = rgb[i][1];
		data[i * 3 + 2] = rgb[i][2];
	}
	delete[] rgb;
}