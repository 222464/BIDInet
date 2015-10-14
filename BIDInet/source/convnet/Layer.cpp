#include "Layer.h"

using namespace convnet;

float convnet::relu(float x, float leak) {
	if (x > 0.0f)
		return x;

	return x * leak;
}

float convnet::relud(float x, float leak) {
	if (x > 0.0f)
		return 1.0f;

	return leak;
}