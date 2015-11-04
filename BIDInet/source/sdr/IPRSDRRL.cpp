#include "IPRSDRRL.h"

#include <SFML/Window.hpp>
#include <iostream>

#include <algorithm>

using namespace sdr;

void IPRSDRRL::createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_inputTypes = inputTypes;

	for (int i = 0; i < _inputTypes.size(); i++) {
		if (_inputTypes[i] == _action)
			_actionInputIndices.push_back(i);
	}

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int widthPrev = inputWidth;
	int heightPrev = inputHeight;

	for (int l = 0; l < _layerDescs.size(); l++) {
		_layers[l]._sdr.createRandom(widthPrev, heightPrev, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._receptiveRadius, _layerDescs[l]._recurrentRadius, initMinWeight, initMaxWeight, generator);

		_layers[l]._predictionNodes.resize(_layerDescs[l]._width * _layerDescs[l]._height);

		int feedBackSize = std::pow(_layerDescs[l]._feedBackRadius * 2 + 1, 2);
		int predictiveSize = std::pow(_layerDescs[l]._predictiveRadius * 2 + 1, 2);

		float hiddenToNextHiddenWidth = 1.0f;
		float hiddenToNextHiddenHeight = 1.0f;

		if (l < _layers.size() - 1) {
			hiddenToNextHiddenWidth = static_cast<float>(_layerDescs[l + 1]._width) / static_cast<float>(_layerDescs[l]._width);
			hiddenToNextHiddenHeight = static_cast<float>(_layerDescs[l + 1]._height) / static_cast<float>(_layerDescs[l]._height);
		}

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			int hx = pi % _layerDescs[l]._width;
			int hy = pi / _layerDescs[l]._width;

			// Feed Back
			if (l < _layers.size() - 1) {
				p._feedBackConnectionIndices.reserve(feedBackSize);

				int centerX = std::round(hx * hiddenToNextHiddenWidth);
				int centerY = std::round(hy * hiddenToNextHiddenHeight);

				for (int dx = -_layerDescs[l]._feedBackRadius; dx <= _layerDescs[l]._feedBackRadius; dx++)
					for (int dy = -_layerDescs[l]._feedBackRadius; dy <= _layerDescs[l]._feedBackRadius; dy++) {
						int hox = centerX + dx;
						int hoy = centerY + dy;

						if (hox >= 0 && hox < _layerDescs[l + 1]._width && hoy >= 0 && hoy < _layerDescs[l + 1]._height) {
							int hio = hox + hoy * _layerDescs[l + 1]._width;

							p._feedBackConnectionIndices.push_back(hio);
						}
					}

				p._feedBackConnectionIndices.shrink_to_fit();
			}

			// Predictive
			p._predictiveConnectionIndices.reserve(feedBackSize);

			for (int dx = -_layerDescs[l]._predictiveRadius; dx <= _layerDescs[l]._predictiveRadius; dx++)
				for (int dy = -_layerDescs[l]._predictiveRadius; dy <= _layerDescs[l]._predictiveRadius; dy++) {
					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < _layerDescs[l]._width && hoy >= 0 && hoy < _layerDescs[l]._height) {
						int hio = hox + hoy * _layerDescs[l]._width;

						p._predictiveConnectionIndices.push_back(hio);
					}
				}

			p._predictiveConnectionIndices.shrink_to_fit();

			p._sdrrl.createRandom(p._feedBackConnectionIndices.size() + p._predictiveConnectionIndices.size(), 1, _layerDescs[l]._cellCount, initMinWeight, initMaxWeight, initMinInhibition, initMaxInhibition, initThreshold, generator);
		}

		widthPrev = _layerDescs[l]._width;
		heightPrev = _layerDescs[l]._height;
	}

	_inputPredictionNodes.resize(_inputTypes.size());

	float inputToNextHiddenWidth = static_cast<float>(_layerDescs.front()._width) / static_cast<float>(inputWidth);
	float inputToNextHiddenHeight = static_cast<float>(_layerDescs.front()._height) / static_cast<float>(inputHeight);

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		PredictionNode &p = _inputPredictionNodes[pi];

		int hx = pi % inputWidth;
		int hy = pi / inputWidth;

		int feedBackSize = std::pow(inputFeedBackRadius * 2 + 1, 2);

		// Feed Back
		p._feedBackConnectionIndices.reserve(feedBackSize);

		int centerX = std::round(hx * inputToNextHiddenWidth);
		int centerY = std::round(hy * inputToNextHiddenHeight);

		for (int dx = -inputFeedBackRadius; dx <= inputFeedBackRadius; dx++)
			for (int dy = -inputFeedBackRadius; dy <= inputFeedBackRadius; dy++) {
				int hox = centerX + dx;
				int hoy = centerY + dy;

				if (hox >= 0 && hox < _layerDescs.front()._width && hoy >= 0 && hoy < _layerDescs.front()._height) {
					int hio = hox + hoy * _layerDescs.front()._width;

					p._feedBackConnectionIndices.push_back(hio);
				}
			}

		p._feedBackConnectionIndices.shrink_to_fit();

		p._sdrrl.createRandom(p._feedBackConnectionIndices.size() + p._predictiveConnectionIndices.size(), 1, _cellCount, initMinWeight, initMaxWeight, initMinInhibition, initMaxInhibition, initThreshold, generator);
	}
}

void IPRSDRRL::simStep(float reward, std::mt19937 &generator) {
	// Feature extraction
	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._sdr.activate(_layerDescs[l]._sdrIter, _layerDescs[l]._sdrStepSize, _layerDescs[l]._sdrLambda, _layerDescs[l]._sdrHiddenDecay, _layerDescs[l]._sdrNoise, generator);

		// Set inputs for next layer if there is one
		if (l < _layers.size() - 1) {
			for (int i = 0; i < _layers[l]._sdr.getNumHidden(); i++)
				_layers[l + 1]._sdr.setVisibleState(i, _layers[l]._sdr.getHiddenState(i));
		}
	}

	// Prediction
	for (int l = _layers.size() - 1; l >= 0; l--) {
		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			// Update action towards prediction a bit
			int stateIndex = 0;

			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnectionIndices.size(); ci++)
					p._sdrrl.setState(stateIndex++, _layers[l + 1]._predictionNodes[p._feedBackConnectionIndices[ci]]._sdrrl.getAction(0));
			}

			// Predictive
			for (int ci = 0; ci < p._predictiveConnectionIndices.size(); ci++)
				p._sdrrl.setState(stateIndex++, _layers[l]._sdr.getHiddenStatePrev(p._predictiveConnectionIndices[ci]));

			p._sdrrl.simStepDrift(std::vector<float>(1, _layers[l]._sdr.getHiddenState(pi)), reward, _layerDescs[l]._cellSparsity, _layerDescs[l]._gamma,
				_layerDescs[l]._gateFeedForwardAlpha, _layerDescs[l]._gateThresholdAlpha,
				_layerDescs[l]._qAlpha, _layerDescs[l]._actionAlpha,
				_layerDescs[l]._gammaLambda, _layerDescs[l]._explorationStdDev, _layerDescs[l]._explorationBreakChance,
				_layerDescs[l]._averageSurpriseDecay, _layerDescs[l]._surpriseLearnFactor, generator);
		}
	}
	
	// Get first layer prediction
	{
		for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
			PredictionNode &p = _inputPredictionNodes[pi];

			// Update action towards prediction a bit
			int stateIndex = 0;

			for (int ci = 0; ci < p._feedBackConnectionIndices.size(); ci++)
				p._sdrrl.setState(stateIndex++, _layers.front()._predictionNodes[p._feedBackConnectionIndices[ci]]._sdrrl.getAction(0));

			p._sdrrl.simStepDrift(std::vector<float>(1, _layers.front()._sdr.getVisibleState(pi)), reward, _cellSparsity, _gamma,
				_gateFeedForwardAlpha, _gateThresholdAlpha,
				_qAlpha, _actionAlpha,
				_gammaLambda, _explorationStdDev, _explorationBreakChance,
				_averageSurpriseDecay, _surpriseLearnFactor, generator);
		}
	}

	// Set inputs to predictions
	for (int i = 0; i < _inputTypes.size(); i++)
		if (_inputTypes[i] == _action)
			_layers.front()._sdr.setVisibleState(i, getAction(i));
}