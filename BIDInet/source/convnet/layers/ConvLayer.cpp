#include "ConvLayer.h"

using namespace convnet;

void ConvLayer::create(Layer &input, int width, int height, int numMaps, int convWidth, int convHeight,
	float initMinWeight, float initMaxWeight, std::mt19937 &generator)
{
	_outputMaps.resize(numMaps);

	_errorMaps.resize(numMaps);

	for (int m = 0; m < _outputMaps.size(); m++)
		_outputMaps[m].create(width, height);

	for (int m = 0; m < _errorMaps.size(); m++)
		_errorMaps[m].create(width, height);
	
	_convWidth = convWidth;
	_convHeight = convHeight;
	_convNumMaps = input.getNumMaps();

	// Create kernels
	_mapKernels.resize(_outputMaps.size());

	int connectionsSize = _convWidth * _convHeight * _convNumMaps;

	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	
	for (int m = 0; m < _outputMaps.size(); m++) {
		_mapKernels[m]._connections.resize(connectionsSize);

		for (int c = 0; c < connectionsSize; c++)
			_mapKernels[m]._connections[c]._weight = weightDist(generator);
	}
}

void ConvLayer::forward(const std::vector<Map> &inputMaps) {
	float outputToInputX = static_cast<float>(inputMaps.front().getWidth()) / _outputMaps.front().getWidth();
	float outputToInputY = static_cast<float>(inputMaps.front().getHeight()) / _outputMaps.front().getHeight();

	int lowerX = std::ceil(-_convWidth * 0.5f);
	int upperX = std::ceil(_convWidth * 0.5f);
	int lowerY = std::ceil(-_convHeight * 0.5f);
	int upperY = std::ceil(_convHeight * 0.5f);

	for (int m = 0; m < _outputMaps.size(); m++) {		
		for (int x = 0; x < _outputMaps[m].getWidth(); x++)
			for (int y = 0; y < _outputMaps[m].getHeight(); y++) {
				float sum = 0.0f;

				int centerX = std::round(outputToInputX * x);
				int centerY = std::round(outputToInputY * y);

				int wi = 0;

				for (int dx = lowerX; dx < upperX; dx++)
					for (int dy = lowerY; dy < upperY; dy++) {
						int xo = centerX + dx;
						int yo = centerY + dy;

						if (xo >= 0 && xo < inputMaps.front().getWidth() && yo >= 0 && yo < inputMaps.front().getHeight()) {
							for (int mo = 0; mo < inputMaps.size(); mo++) {
								sum += _mapKernels[m]._connections[wi]._weight * inputMaps[mo].atXY(xo, yo);

								wi++;
							}
						}
						else
							wi += inputMaps.size();
					}

				_outputMaps[m].atXY(x, y) = relu(sum, _reluLeak);
			}
	}
}

void ConvLayer::backward(std::vector<Map> &errorMaps) {
	float outputToInputX = static_cast<float>(errorMaps.front().getWidth()) / _outputMaps.front().getWidth();
	float outputToInputY = static_cast<float>(errorMaps.front().getHeight()) / _outputMaps.front().getHeight();

	int lowerX = std::ceil(-_convWidth * 0.5f);
	int upperX = std::ceil(_convWidth * 0.5f);
	int lowerY = std::ceil(-_convHeight * 0.5f);
	int upperY = std::ceil(_convHeight * 0.5f);

	for (int m = 0; m < errorMaps.size(); m++)
		errorMaps[m].clear(0.0f);

	for (int m = 0; m < _outputMaps.size(); m++) {
		for (int x = 0; x < _outputMaps[m].getWidth(); x++)
			for (int y = 0; y < _outputMaps[m].getHeight(); y++) {
				int centerX = std::round(outputToInputX * x);
				int centerY = std::round(outputToInputY * y);

				float error = _errorMaps[m].atXY(x, y) * relud(_outputMaps[m].atXY(x, y), _reluLeak);

				int wi = 0;

				for (int dx = lowerX; dx < upperX; dx++)
					for (int dy = lowerY; dy < upperY; dy++) {
						int xo = centerX + dx;
						int yo = centerY + dy;

						if (xo >= 0 && xo < errorMaps.front().getWidth() && yo >= 0 && yo < errorMaps.front().getHeight()) {
							for (int mo = 0; mo < errorMaps.size(); mo++) {
								errorMaps[mo].atXY(xo, yo) += _mapKernels[m]._connections[wi]._weight * error;
			
								wi++;
							}
						}
						else
							wi += errorMaps.size();
					}
			}
	}
}

void ConvLayer::update(const std::vector<Map> &inputMaps) {
	// Update kernels
	float outputToInputX = static_cast<float>(inputMaps.front().getWidth()) / _outputMaps.front().getWidth();
	float outputToInputY = static_cast<float>(inputMaps.front().getHeight()) / _outputMaps.front().getHeight();

	int lowerX = std::ceil(-_convWidth * 0.5f);
	int upperX = std::ceil(_convWidth * 0.5f);
	int lowerY = std::ceil(-_convHeight * 0.5f);
	int upperY = std::ceil(_convHeight * 0.5f);

	for (int m = 0; m < _outputMaps.size(); m++) {
		for (int x = 0; x < _outputMaps[m].getWidth(); x++)
			for (int y = 0; y < _outputMaps[m].getHeight(); y++) {
				int centerX = std::round(outputToInputX * x);
				int centerY = std::round(outputToInputY * y);

				float error = _errorMaps[m].atXY(x, y) * relud(_outputMaps[m].atXY(x, y), _reluLeak);

				int wi = 0;

				for (int dx = lowerX; dx < upperX; dx++)
					for (int dy = lowerY; dy < upperY; dy++) {
						int xo = centerX + dx;
						int yo = centerY + dy;

						if (xo >= 0 && xo < inputMaps.front().getWidth() && yo >= 0 && yo < inputMaps.front().getHeight()) {
							for (int mo = 0; mo < inputMaps.size(); mo++) {
								float delta = _alpha * error * inputMaps[mo].atXY(xo, yo);

								_mapKernels[m]._connections[wi]._weight += delta;
	
								wi++;
							}
						}
						else
							wi += inputMaps.size();
					}
			}
	}
}