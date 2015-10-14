#include "MaxPoolingLayer.h"

#include <algorithm>

#include <assert.h>

using namespace convnet;

void MaxPoolingLayer::create(Layer &input, int width, int height, int numMaps, int poolWidth, int poolHeight) {
	_outputMaps.resize(numMaps);

	_errorMaps.resize(numMaps);

	_maxIndices.resize(numMaps);

	for (int m = 0; m < _outputMaps.size(); m++)
		_outputMaps[m].create(width, height);

	for (int m = 0; m < _errorMaps.size(); m++)
		_errorMaps[m].create(width, height);

	for (int m = 0; m < _maxIndices.size(); m++)
		_maxIndices[m].create(width, height);

	_poolWidth = poolWidth;
	_poolHeight = poolHeight;
}

void MaxPoolingLayer::forward(const std::vector<Map> &inputMaps) {
	float outputToInputX = static_cast<float>(inputMaps.front().getWidth()) / _outputMaps.front().getWidth();
	float outputToInputY = static_cast<float>(inputMaps.front().getHeight()) / _outputMaps.front().getHeight();

	int lowerX = std::ceil(-_poolWidth * 0.5f);
	int upperX = std::ceil(_poolWidth * 0.5f);
	int lowerY = std::ceil(-_poolHeight * 0.5f);
	int upperY = std::ceil(_poolHeight * 0.5f);

	for (int m = 0; m < _outputMaps.size(); m++) {
		for (int x = 0; x < _outputMaps[m].getWidth(); x++)
			for (int y = 0; y < _outputMaps[m].getHeight(); y++) {
				float pool = -999999.0f;

				int centerX = std::round(outputToInputX * x);
				int centerY = std::round(outputToInputY * y);

				int wi = 0;

				for (int dx = lowerX; dx < upperX; dx++)
					for (int dy = lowerY; dy < upperY; dy++) {
						int xo = centerX + dx;
						int yo = centerY + dy;

						if (xo >= 0 && xo < inputMaps.front().getWidth() && yo >= 0 && yo < inputMaps.front().getHeight()) {
							for (int mo = 0; mo < inputMaps.size(); mo++) {
								if (inputMaps[mo].atXY(xo, yo) > pool) {
									pool = inputMaps[mo].atXY(xo, yo);

									_maxIndices[m].atXY(x, y) = wi;
								}

								wi++;
							}
						}
						else
							wi += inputMaps.size();
					}

				assert(pool != -999999.0f);

				_outputMaps[m].atXY(x, y) = pool;
			}
	}
}

void MaxPoolingLayer::backward(std::vector<Map> &errorMaps) {
	float outputToInputX = static_cast<float>(errorMaps.front().getWidth()) / _outputMaps.front().getWidth();
	float outputToInputY = static_cast<float>(errorMaps.front().getHeight()) / _outputMaps.front().getHeight();

	int lowerX = std::ceil(-_poolWidth * 0.5f);
	int upperX = std::ceil(_poolWidth * 0.5f);
	int lowerY = std::ceil(-_poolHeight * 0.5f);
	int upperY = std::ceil(_poolHeight * 0.5f);

	for (int m = 0; m < errorMaps.size(); m++)
		errorMaps[m].clear(0.0f);

	for (int m = 0; m < _outputMaps.size(); m++) {
		for (int x = 0; x < _outputMaps[m].getWidth(); x++)
			for (int y = 0; y < _outputMaps[m].getHeight(); y++) {
				int centerX = std::round(outputToInputX * x);
				int centerY = std::round(outputToInputY * y);

				float error = _errorMaps[m].atXY(x, y);

				int wi = 0;

				for (int dx = lowerX; dx < upperX; dx++)
					for (int dy = lowerY; dy < upperY; dy++) {
						int xo = centerX + dx;
						int yo = centerY + dy;

						if (xo >= 0 && xo < errorMaps.front().getWidth() && yo >= 0 && yo < errorMaps.front().getHeight()) {
							for (int mo = 0; mo < errorMaps.size(); mo++) {
								if (wi == _maxIndices[m].atXY(x, y))
									errorMaps[mo].atXY(xo, yo) += error;

								wi++;
							}
						}
						else
							wi += errorMaps.size();
					}
			}
	}
}