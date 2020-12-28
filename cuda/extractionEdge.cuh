#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

unsigned char *edge(unsigned char *data, int width, int hight);