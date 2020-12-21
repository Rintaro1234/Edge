#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

uchar *edge(uchar *data, int width, int hight);