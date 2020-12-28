#include "extractionEdge.cuh"

// 一ブロック当たりのスレッド数
#define threads 256

#define edgePercentage (unsigned char)(0.09 * 255)

texture<unsigned char, cudaTextureType1D> tex;

// カーネル
__global__ void kernel(int width, int hight, unsigned char *output)
{
	// CUDA処理を書く
	//自分のスレッドのindex
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// 範囲からはみ出るようなら逃げる
	if (idx >= width) return;

	//テスト処理(白黒化)
	for (int i = 0; i < hight; i++)
	{
		int targetIdx	= idx * 3 + width * i * 3;
		
		// 中心の値を取得
		unsigned char tr = tex1Dfetch(tex, targetIdx + 0);
		unsigned char tg = tex1Dfetch(tex, targetIdx + 1);
		unsigned char tb = tex1Dfetch(tex, targetIdx + 2);

		//左側
		unsigned char lr = (threadIdx.x <= 0) ? tex1Dfetch(tex, targetIdx + 3) : tex1Dfetch(tex, targetIdx - 3);
		unsigned char averageLr = abs(lr - tr);
		unsigned char lg = (threadIdx.x <= 0) ? tex1Dfetch(tex, targetIdx + 4) : tex1Dfetch(tex, targetIdx - 2);
		unsigned char averageLg = abs(lg - tg);
		unsigned char lb = (threadIdx.x <= 0) ? tex1Dfetch(tex, targetIdx + 5) : tex1Dfetch(tex, targetIdx - 1);
		unsigned char averageLb = abs(lb - tb);

		// 差が一番大きいものを使用
		unsigned char averageLeft = (averageLr < averageLg) ?
									(averageLg < averageLb) ? averageLb : averageLg
								   :(averageLr < averageLb) ? averageLb : averageLr;

		//右側
		unsigned char rr = (threadIdx.x <= width) ? tex1Dfetch(tex, targetIdx + 3) : tex1Dfetch(tex, targetIdx - 3);
		unsigned char averageRr = abs(rr - tr);
		unsigned char rg = (threadIdx.x <= width) ? tex1Dfetch(tex, targetIdx + 4) : tex1Dfetch(tex, targetIdx - 2);
		unsigned char averageRg = abs(rg - tg);
		unsigned char rb = (threadIdx.x <= width) ? tex1Dfetch(tex, targetIdx + 5) : tex1Dfetch(tex, targetIdx - 1);
		unsigned char averageRb = abs(rb - tb);

		// 差が一番大きいものを使用
		unsigned char averageRight = (averageRr < averageRg) ?
									 (averageRg < averageRb) ? averageRb : averageRg
									:(averageRr < averageRb) ? averageRb : averageRr;

		// 上側
		unsigned char ur = (i <= 0) ? tex1Dfetch(tex, targetIdx + width * 3 + 0) : tex1Dfetch(tex, targetIdx - width * 3 + 0);
		unsigned char averageUr = abs(ur - tr);
		unsigned char ug = (i <= 0) ? tex1Dfetch(tex, targetIdx + width * 3 + 1) : tex1Dfetch(tex, targetIdx - width * 3 + 1);
		unsigned char averageUg = abs(ug - tg);
		unsigned char ub = (i <= 0) ? tex1Dfetch(tex, targetIdx + width * 3 + 2) : tex1Dfetch(tex, targetIdx - width * 3 + 2);
		unsigned char averageUb = abs(ub - tb);

		// 差が一番大きいものを使用
		unsigned char averageUp = (averageUr < averageUg) ?
								  (averageUg < averageUb) ? averageUb : averageUg
								 :(averageUr < averageUb) ? averageUb : averageUr;

		// 下側
		unsigned char dr = (i < hight) ? tex1Dfetch(tex, targetIdx + width * 3 + 0) : tex1Dfetch(tex, targetIdx - width * 3 + 0);
		unsigned char averageDr = abs(dr - tr);
		unsigned char dg = (i < hight) ? tex1Dfetch(tex, targetIdx + width * 3 + 1) : tex1Dfetch(tex, targetIdx - width * 3 + 1);
		unsigned char averageDg = abs(dg - tg);
		unsigned char db = (i < hight) ? tex1Dfetch(tex, targetIdx + width * 3 + 2) : tex1Dfetch(tex, targetIdx - width * 3 + 2);
		unsigned char averageDb = abs(db - tb);

		// 差が一番大きいものを使用
		unsigned char averageDown = (averageDr < averageDg) ?
									(averageDg < averageDb) ? averageDb : averageDg
								   :(averageDr < averageDb) ? averageDb : averageDr;

		unsigned char difference = (averageLeft + averageRight + averageUp + averageDown) / 4.0f;
		output[idx + width * i] = (edgePercentage < difference) ? 0 : 255;;
	}

	return;
}

unsigned char *edge(unsigned char *data, int width, int hight)
{
	// 画像のサイズ
	int length = width * hight;
	size_t pixelSize = sizeof(unsigned char) * length;
	size_t dataSize = pixelSize * 3;

	// ホスト側のポインタ
	unsigned char *pHostOutput;

	// デバイス側のポインタ
	unsigned char *pDevData;
	unsigned char *pDevOutput;

	// メモリの確保
	cudaMallocHost(&pHostOutput, pixelSize);
	cudaMalloc(&pDevData, dataSize);
	cudaMalloc(&pDevOutput, pixelSize);

	// テクスチャメモリにバインド
	auto desc = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture(NULL, &tex, pDevData, &desc, dataSize);

	// データの転送
	cudaMemcpy(pDevData, data, dataSize, cudaMemcpyHostToDevice);

	// カーネルの実行
	dim3 block(threads, 1, 1);
	dim3 grid((width + threads - 1) / threads, 1, 1);
	kernel <<< grid, block >>> (width, hight, pDevOutput);

	// データの転送
	cudaMemcpy(pHostOutput, pDevOutput, pixelSize, cudaMemcpyDeviceToHost);

	// バインド解除
	cudaUnbindTexture(tex);

	// デバイスメモリの解放とリセット
	cudaFree(pDevData);
	cudaFree(pDevOutput);

	//cudaDeviceReset();

	return pHostOutput;
}