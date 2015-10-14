#include "ConvNet.h"

using namespace convnet;

void ConvNet::create(int numHidden, int numOutputs, float initMinWeight, float initMaxWeight, std::mt19937 &generator) {
	_hiddenNodes.resize(numHidden);
	_outputNodes.resize(numOutputs);

	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	int numHiddenInputs = _layers.back()->getOutputWidth() * _layers.back()->getOutputHeight() * _layers.back()->getNumMaps();

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		_hiddenNodes[h]._connections.resize(numHiddenInputs);

		for (int c = 0; c < _hiddenNodes[h]._connections.size(); c++)
			_hiddenNodes[h]._connections[c]._weight = weightDist(generator);

		_hiddenNodes[h]._bias._weight = weightDist(generator);
	}

	for (int h = 0; h < _outputNodes.size(); h++) {
		_outputNodes[h]._connections.resize(_hiddenNodes.size());

		for (int c = 0; c < _outputNodes[h]._connections.size(); c++)
			_outputNodes[h]._connections[c]._weight = weightDist(generator);

		_outputNodes[h]._bias._weight = weightDist(generator);
	}
}

void ConvNet::forward() {
	// First layer is input, so skip that one
	for (int l = 1; l < _layers.size(); l++)
		_layers[l]->forward(_layers[l - 1]->getOutputMaps());

	int lastMapSize = _layers.back()->getOutputWidth() * _layers.back()->getOutputHeight();

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float sum = _hiddenNodes[h]._bias._weight;

		int wi = 0;

		for (int c = 0; c < lastMapSize; c++)
			for (int m = 0; m < _layers.back()->getOutputMaps().size(); m++) {
				sum += _hiddenNodes[h]._connections[wi]._weight * _layers.back()->getOutputMaps()[m][c];

				wi++;
			}

		_hiddenNodes[h]._output = relu(sum, _reluLeak);
	}

	for (int h = 0; h < _outputNodes.size(); h++) {
		float sum = _outputNodes[h]._bias._weight;

		for (int c = 0; c < _outputNodes[h]._connections.size(); c++)
			sum += _outputNodes[h]._connections[c]._weight * _hiddenNodes[c]._output;

		_outputNodes[h]._output = sum;
	}
}

void ConvNet::backward() {
	// Propagate to hidden layer
	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float sum = 0.0f;

		for (int i = 0; i < _outputNodes.size(); i++)
			sum += _outputNodes[i]._connections[h]._weight * _outputNodes[i]._error;
		
		_hiddenNodes[h]._error = sum * relud(_hiddenNodes[h]._output, _reluLeak);
	}

	int lastMapSize = _layers.back()->getOutputWidth() * _layers.back()->getOutputHeight();

	// Propagate to first layer
	for(int m = 0; m < _layers.back()->getOutputMaps().size(); m++)
		_layers.back()->getOutputMaps()[m].clear(0.0f);

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		int wi = 0;

		for (int c = 0; c < lastMapSize; c++)
			for (int m = 0; m < _layers.back()->getOutputMaps().size(); m++) {
				_layers.back()->getErrorMaps()[m][c] += _hiddenNodes[h]._connections[wi]._weight * _hiddenNodes[h]._error;

				wi++;
			}
	}

	// Propagate rest of layers except for first two
	for (int l = _layers.size() - 1; l > 1; l--)
		_layers[l]->backward(_layers[l - 1]->getErrorMaps());
}

void ConvNet::update() {
	for (int h = 0; h < _outputNodes.size(); h++) {
		for (int c = 0; c < _outputNodes[h]._connections.size(); c++) {
			float delta = _outputAlpha * _outputNodes[h]._error * _hiddenNodes[c]._output + _outputMomentum * _outputNodes[h]._connections[c]._prevDWeight;

			_outputNodes[h]._connections[c]._weight += delta;
			_outputNodes[h]._connections[c]._prevDWeight = delta;
		}

		float delta = _outputAlpha * _outputNodes[h]._error + _outputMomentum * _outputNodes[h]._bias._prevDWeight;

		_outputNodes[h]._bias._weight += delta;
		_outputNodes[h]._bias._prevDWeight = delta;
	}

	int lastMapSize = _layers.back()->getOutputWidth() * _layers.back()->getOutputHeight();

	for (int h = 0; h < _hiddenNodes.size(); h++) {
		float sum = _hiddenNodes[h]._bias._weight;

		int wi = 0;

		for (int c = 0; c < lastMapSize; c++)
			for (int m = 0; m < _layers.back()->getOutputMaps().size(); m++) {
				float delta = _hiddenAlpha * _hiddenNodes[h]._error * _layers.back()->getOutputMaps()[m][c] + _hiddenMomentum * _hiddenNodes[h]._connections[wi]._prevDWeight;

				_hiddenNodes[h]._connections[wi]._weight += delta;
				_hiddenNodes[h]._connections[wi]._prevDWeight = delta;

				wi++;
			}

		float delta = _outputAlpha * _hiddenNodes[h]._error + _outputMomentum * _hiddenNodes[h]._bias._prevDWeight;

		_hiddenNodes[h]._bias._weight += delta;
		_hiddenNodes[h]._bias._prevDWeight = delta;
	}

	for (int l = _layers.size() - 1; l > 0; l--)
		_layers[l]->update(_layers[l - 1]->getOutputMaps());
}