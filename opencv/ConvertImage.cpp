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

// �摜�f�[�^�[����G�b�W�����o
void ConvertImage::edge(uchar *data, int width, int hight, uchar *output)
{
	int pixels = width * hight;

	// �G�b�W���o
	#pragma omp parallel for
	for (int y = 0; y < hight; y++)
	{
		for (int x = 0; x < width; x++)
		{
			// ���ꂼ��̔z��̈ʒu���Z�o
			int target	= (x + y * width) * 3;
			int left	= target - 3;
			int right	= target + 3;
			int up		= target - width * 3;
			int down	= target + width * 3;

			int sideMin = (width * 3) * y;
			int sideMax = (width * 3) * (y + 1);
			int downMax = (width * 3 * hight) + x * 3;

			// �^��
			int tr = data[target + 0];
			int tg = data[target + 1];
			int tb = data[target + 2];

			// ��
			int lr = (left < sideMin) ? data[right + 0] : data[left + 0];
			int lg = (left < sideMin) ? data[right + 1] : data[left + 1];
			int lb = (left < sideMin) ? data[right + 2] : data[left + 2];

			// �����Ƃ̍��̊���
			float averageLr = abs(lr - tr) / 255.0f;
			float averageLg = abs(lg - tg) / 255.0f;
			float averageLb = abs(lb - tb) / 255.0f;
			// ��ԍ����傫�����̂��Ƃ�
			float averageLeft =	 (averageLr < averageLg) ?
								 (averageLg < averageLb) ? averageLb : averageLg 
								:(averageLr < averageLb) ? averageLb : averageLr;

			// �E
			int rr = (sideMax <= right) ? data[left + 0] : data[right + 0];
			int rg = (sideMax <= right) ? data[left + 1] : data[right + 1];
			int rb = (sideMax <= right) ? data[left + 2] : data[right + 2];

			// �E���Ƃ̍��̊���
			float averageRr = abs(rr - tr) / 255.0f;
			float averageRg = abs(rg - tg) / 255.0f;
			float averageRb = abs(rb - tb) / 255.0f;
			// ��ԍ����傫�����̂��Ƃ�
			float averageRight =	 (averageRr < averageRg) ? 
									 (averageRg < averageRb) ? averageRb : averageRg 
									:(averageRr < averageRb) ? averageRb : averageRr;

			// ���E�ʂƂ̍��̕���
			float differenceS = (averageLeft + averageRight) / 2.0f;

			// ��
			int ur = (up < 0) ? data[down + 0] : data[up + 0];
			int ug = (up < 0) ? data[down + 1] : data[up + 1];
			int ub = (up < 0) ? data[down + 2] : data[up + 2];

			// �㑤�Ƃ̍��̊���
			float averageUr = abs(ur - tr) / 255.0f;
			float averageUg = abs(ug - tg) / 255.0f;
			float averageUb = abs(ub - tb) / 255.0f;
			// ��ԍ����傫�����̂��Ƃ�
			float averageUp =	 (averageUr < averageUg) ? 
								 (averageUg < averageUb) ? averageUb : averageUg
								:(averageUr < averageUb) ? averageUb : averageUr;

			// ��
			int dr = (downMax <= down) ? data[up + 0] : data[down + 0];
			int dg = (downMax <= down) ? data[up + 1] : data[down + 1];
			int db = (downMax <= down) ? data[up + 2] : data[down + 2];

			// �����Ƃ̍��̊���
			float averageDr = abs(dr - tr) / 255.0f;
			float averageDg = abs(dg - tg) / 255.0f;
			float averageDb = abs(db - tb) / 255.0f;
			// ��ԍ����傫�����̂��Ƃ�
			float averageDown =	 (averageDr < averageDg) ? 
								 (averageDg < averageDb) ? averageDb : averageDg
								:(averageDr < averageDb) ? averageDb : averageDr;
			
			// �㉺�ʂƂ̍��̕���
			float differenceV = (averageUp + averageDown) / 2.0f;

			// ����
			float difference = differenceV + differenceS;
			float d = (0.18f < difference) ? 0.0f : 1.0f;
			output[x + y * width] = (uchar)(d * 255);
		}
	}
}