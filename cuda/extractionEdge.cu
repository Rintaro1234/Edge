#include "extractionEdge.cuh"

//float extractionEdge::edgePercentage = 0.09f;

// カーネル
__global__ void kernel(uchar *data, int width, int hight, uchar *output)
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

		uchar average = (data[target + 0] + data[target + 1] + data[target + 2]) * 0.333;
		output[idx + width * i] = average;
	}

	return;
}

//void extractionEdge::edge(uchar *data, int width, int hight, uchar *output)
uchar *edge(uchar *data, int width, int hight)
{
	// 画像のサイズ
	int length = width * hight;
	size_t size = sizeof(uchar) * length;

	// ホスト側のポインタ
	uchar *output;

	// デバイス側のポインタ
	uchar *pDevData;
	uchar *pDevOutput;

	// メモリの確保
	cudaMallocHost(&output, size);
	cudaMalloc(&pDevData, size * 3);
	cudaMalloc(&pDevOutput, size);

	// データの転送
	cudaMemcpy(pDevData, data, size * 3, cudaMemcpyHostToDevice);

	// カーネルの実行
	dim3 block(128, 1, 1);
	dim3 grid((width + 128 - 1) / 128, 1, 1);
	kernel <<< grid, block >>> (pDevData, width, hight, pDevOutput);

	// データの転送
	cudaMemcpy(output, pDevOutput, size, cudaMemcpyDeviceToHost);

	// デバイスメモリの解放とリセット
	cudaFree(pDevData);
	cudaFree(pDevOutput);

	//cudaDeviceReset();

	return output;
}