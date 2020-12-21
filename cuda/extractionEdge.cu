#include "extractionEdge.cuh"

//float extractionEdge::edgePercentage = 0.09f;

texture<uchar, cudaTextureType1D> tex;

// カーネル
__global__ void kernel(int width, int hight, uchar *output)
{
	// CUDA処理を書く
	//自分のスレッドのindex
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// 範囲からはみ出るようなら逃げる
	if (idx >= width) return;

	//テスト処理(白黒化)
	for (int i = 0; i < hight; i++)
	{
		int target = idx * 3 + width * i * 3;
		uchar average = (tex1Dfetch(tex, target + 0) + tex1Dfetch(tex, target + 1) + tex1Dfetch(tex, target + 2)) * 0.333;
		output[idx + width * i] = average;
	}

	return;
}

//void extractionEdge::edge(uchar *data, int width, int hight, uchar *output)
uchar *edge(uchar *data, int width, int hight)
{
	// 画像のサイズ
	int length = width * hight;
	size_t pixelSize = sizeof(uchar) * length;
	size_t dataSize = sizeof(uchar) * length * 3;

	// ホスト側のポインタ
	uchar *pHostOutput;

	// デバイス側のポインタ
	uchar *pDevData;
	uchar *pDevOutput;

	// メモリの確保
	cudaMallocHost(&pHostOutput, pixelSize);
	cudaMalloc(&pDevData, dataSize);
	cudaMalloc(&pDevOutput, pixelSize);

	// テクスチャメモリにバインド
	auto desc = cudaCreateChannelDesc<uchar>();
	cudaBindTexture(NULL, &tex, pDevData, &desc, dataSize);

	// データの転送
	cudaMemcpy(pDevData, data, dataSize, cudaMemcpyHostToDevice);

	// カーネルの実行
	dim3 block(128, 1, 1);
	dim3 grid((width + 128 - 1) / 128, 1, 1);
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