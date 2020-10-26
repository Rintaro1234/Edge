#include "extractionEdge.h"

// ���O�o�b�t�@�[
extern std::stringstream logBuf;

float extractionEdge::edgePercentage = 0.09f;

// �摜�f�[�^�[����G�b�W�����o
void extractionEdge::edge(uchar *data, int width, int hight, uchar *output)
{
	int pixels = width * hight;
	float previousAverageR = 0.0f;

	// �G�b�W���o
	#pragma omp parallel for
	for (int y = 0; y < hight; y++)
	{
		for (int x = 0; x < width; x++)
		{
			// �㉺���E�̔z��ԍ����擾
			int target = (x + y * width) * 3;
			int left = target - 3;
			int right = target + 3;
			int up = target - width * 3;
			int down = target + width * 3;

			int sideMin = (width * 3) * y;
			int sideMax = (width * 3) * (y + 1);
			int downMax = (width * 3 * hight) + x * 3;

			// �^�[�Q�b�g�s�N�Z����RGB�l���擾
			int tr = data[target + 0];
			int tg = data[target + 1];
			int tb = data[target + 2];

			// ����
			int lr = (left < sideMin) ? data[right + 0] : data[left + 0];
			int lg = (left < sideMin) ? data[right + 1] : data[left + 1];
			int lb = (left < sideMin) ? data[right + 2] : data[left + 2];

			// RGB�̍����擾
			float averageLr = abs(lr - tr) / 255.0f;
			float averageLg = abs(lg - tg) / 255.0f;
			float averageLb = abs(lb - tb) / 255.0f;
			// ������ԑ傫�����̂��g�p
			float averageLeft =  (averageLr < averageLg) ?
								 (averageLg < averageLb) ? averageLb : averageLg
								:(averageLr < averageLb) ? averageLb : averageLr;

			// �E��
			int rr = (sideMax <= right) ? data[left + 0] : data[right + 0];
			int rg = (sideMax <= right) ? data[left + 1] : data[right + 1];
			int rb = (sideMax <= right) ? data[left + 2] : data[right + 2];

			// RGB�̍����擾
			float averageRr = abs(rr - tr) / 255.0f;
			float averageRg = abs(rg - tg) / 255.0f;
			float averageRb = abs(rb - tb) / 255.0f;
			// ������ԑ傫�����̂��g�p
			float averageRight =	 (averageRr < averageRg) ?
									 (averageRg < averageRb) ? averageRb : averageRg
									:(averageRr < averageRb) ? averageRb : averageRr;

			// �㑤
			int ur = (up < 0) ? data[down + 0] : data[up + 0];
			int ug = (up < 0) ? data[down + 1] : data[up + 1];
			int ub = (up < 0) ? data[down + 2] : data[up + 2];

			// RGB�̍����擾
			float averageUr = abs(ur - tr) / 255.0f;
			float averageUg = abs(ug - tg) / 255.0f;
			float averageUb = abs(ub - tb) / 255.0f;
			// ��ԍ����傫�����̂��g�p
			float averageUp =	 (averageUr < averageUg) ?
								 (averageUg < averageUb) ? averageUb : averageUg
								:(averageUr < averageUb) ? averageUb : averageUr;

			// ����
			int dr = (downMax <= down) ? data[up + 0] : data[down + 0];
			int dg = (downMax <= down) ? data[up + 1] : data[down + 1];
			int db = (downMax <= down) ? data[up + 2] : data[down + 2];

			// RGB�̍����擾
			float averageDr = abs(dr - tr) / 255.0f;
			float averageDg = abs(dg - tg) / 255.0f;
			float averageDb = abs(db - tb) / 255.0f;
			// ��ԍ����傫�����̂��g�p
			float averageDown =		 (averageDr < averageDg) ?
									 (averageDg < averageDb) ? averageDb : averageDg
									:(averageDr < averageDb) ? averageDb : averageDr;

			// ���ς��擾
			float difference = (averageLeft + averageRight + averageUp + averageDown) / 4.0f;
			float d = (edgePercentage < difference) ? 0.0f : 1.0f;
			output[x + y * width] = (uchar)(d * 255);
		}
	}
}

void extractionEdge::configRead(FILE *cfg)
{
	// �R���t�B�O�t�@�C���̓ǂݍ���
	char tmp[256];
	while (fgets(tmp, 256, cfg) != NULL)
	{
		char *text;
		// �F�̍��̋��e�l���擾
		text = strstr(tmp, "edgePercentage");
		if (text != NULL)
		{
			float data;
			sscanf_s(text, "%*[^0123456789]%f", &data);
			edgePercentage = data;
			goto GET_edgePercentage;
		}

		// ���ׂĂ̗v�f�̓ǂݎ��I��
		goto PASS_thisLine;

	GET_edgePercentage:
		logBuf << "INFO : GET_edgePercentage:" << edgePercentage << std::endl;

	PASS_thisLine:
		logBuf << "INFO : PASS_thisLine" << std::endl;
	}
	logBuf << "INFO : FINISH_ReadConfig" << std::endl;
}