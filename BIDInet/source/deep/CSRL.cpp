#include "CSRL.h"

#include <algorithm>

#include <SFML/Window.hpp>

#include <iostream>

using namespace deep;

void CSRL::createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initBoost, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int widthPrev = inputWidth;
	int heightPrev = inputHeight;

	for (int l = 0; l < _layerDescs.size(); l++) {
		_layers[l]._sdr.createRandom(widthPrev, heightPrev, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._receptiveRadius, _layerDescs[l]._recurrentRadius, initMinWeight, initMaxWeight, initBoost, generator);

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

			p._bias._weight = weightDist(generator);

			int hx = pi % _layerDescs[l]._width;
			int hy = pi / _layerDescs[l]._width;

			// Feed Back
			if (l < _layers.size() - 1) {
				p._feedBackConnections.reserve(feedBackSize);

				int centerX = std::round(hx * hiddenToNextHiddenWidth);
				int centerY = std::round(hy * hiddenToNextHiddenHeight);

				for (int dx = -_layerDescs[l]._feedBackRadius; dx <= _layerDescs[l]._feedBackRadius; dx++)
					for (int dy = -_layerDescs[l]._feedBackRadius; dy <= _layerDescs[l]._feedBackRadius; dy++) {
						int hox = centerX + dx;
						int hoy = centerY + dy;

						if (hox >= 0 && hox < _layerDescs[l + 1]._width && hoy >= 0 && hoy < _layerDescs[l + 1]._height) {
							int hio = hox + hoy * _layerDescs[l + 1]._width;

							Connection c;

							c._weight = weightDist(generator);
							c._index = hio;

							p._feedBackConnections.push_back(c);
						}
					}

				p._feedBackConnections.shrink_to_fit();
			}

			// Predictive
			p._predictiveConnections.reserve(feedBackSize);

			for (int dx = -_layerDescs[l]._predictiveRadius; dx <= _layerDescs[l]._predictiveRadius; dx++)
				for (int dy = -_layerDescs[l]._predictiveRadius; dy <= _layerDescs[l]._predictiveRadius; dy++) {
					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < _layerDescs[l]._width && hoy >= 0 && hoy < _layerDescs[l]._height) {
						int hio = hox + hoy * _layerDescs[l]._width;

						Connection c;

						c._weight = weightDist(generator);
						c._index = hio;

						p._predictiveConnections.push_back(c);
					}
				}

			p._predictiveConnections.shrink_to_fit();

			p._sdrrl.createRandom(p._predictiveConnections.size() + p._feedBackConnections.size(), 2, _layerDescs[l]._cellsPerColumn, initMinWeight, initMaxWeight, 0.0f, generator);
		}

		widthPrev = _layerDescs[l]._width;
		heightPrev = _layerDescs[l]._height;
	}

	_inputPredictionNodes.resize(inputWidth * inputHeight);

	float inputToNextHiddenWidth = static_cast<float>(_layerDescs.front()._width) / static_cast<float>(inputWidth);
	float inputToNextHiddenHeight = static_cast<float>(_layerDescs.front()._height) / static_cast<float>(inputHeight);

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		p._bias._weight = weightDist(generator);

		int hx = pi % inputWidth;
		int hy = pi / inputWidth;

		int feedBackSize = std::pow(inputFeedBackRadius * 2 + 1, 2);

		// Feed Back
		p._feedBackConnections.reserve(feedBackSize);

		int centerX = std::round(hx * inputToNextHiddenWidth);
		int centerY = std::round(hy * inputToNextHiddenHeight);

		for (int dx = -inputFeedBackRadius; dx <= inputFeedBackRadius; dx++)
			for (int dy = -inputFeedBackRadius; dy <= inputFeedBackRadius; dy++) {
				int hox = centerX + dx;
				int hoy = centerY + dy;

				if (hox >= 0 && hox < _layerDescs.front()._width && hoy >= 0 && hoy < _layerDescs.front()._height) {
					int hio = hox + hoy * _layerDescs.front()._width;

					Connection c;

					c._weight = weightDist(generator);
					c._index = hio;

					p._feedBackConnections.push_back(c);
				}
			}

		p._feedBackConnections.shrink_to_fit();

		p._sdrrl.createRandom(p._feedBackConnections.size(), 1, _cellsPerColumn, initMinWeight, initMaxWeight, 0.0f, generator);
	}
}

void CSRL::simStep(float reward, std::mt19937 &generator, bool learn) {
	// Feature extraction
	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._sdr.activate(_layerDescs[l]._sdrIter, _layerDescs[l]._sdrStepSize, _layerDescs[l]._sdrHiddenDecay, _layerDescs[l]._sdrNoise, generator);

		// Set inputs for next layer if there is one
		if (l < _layers.size() - 1) {
			for (int i = 0; i < _layers[l]._sdr.getNumHidden(); i++) {
				// Attention gate
				float gated = _layers[l]._sdr.getHiddenState(i) * _layers[l]._predictionNodes[i]._sdrrl.getAction(2);

				_layers[l + 1]._sdr.setVisibleState(i, gated);
			}
		}
	}

	// Prediction
	std::vector<std::vector<float>> predictionErrors(_layers.size());

	for (int l = _layers.size() - 1; l >= 0; l--) {
		predictionErrors[l].resize(_layers[l]._predictionNodes.size());

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			// Activate SDRRL
			//if (_layers[l]._sdr.getHiddenState(pi) != 0.0f) {
				// Set inputs (lateral and feed back)
				int inputIndex = 0;

				if (l < _layers.size() - 1) {
					for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
						p._sdrrl.setState(inputIndex++, _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._state);
				}

				// Predictive
				for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
					p._sdrrl.setState(inputIndex++, _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index));

				p._sdrrl.simStep(reward, _layerDescs[l]._cellSparsity, _layerDescs[l]._gamma,
					_layerDescs[l]._gateSolveIter, _layerDescs[l]._gateFeedForwardAlpha, _layerDescs[l]._gateThresholdAlpha,
					_layerDescs[l]._qAlpha, _layerDescs[l]._actionAlpha, _layerDescs[l]._actionDeriveIterations, _layerDescs[l]._actionDeriveAlpha,
					_layerDescs[l]._gammaLambda, _layerDescs[l]._explorationStdDev, _layerDescs[l]._explorationBreak,
					_layerDescs[l]._averageSurpriseDecay, _layerDescs[l]._surpriseLearnFactor, generator);
			//}

			// Learn
			if (learn) {
				float predictionError = p._sdrrl.getAction(0) * (_layers[l]._sdr.getHiddenState(pi) - p._statePrev);
				
				predictionErrors[l][pi] = _layers[l]._sdr.getHiddenState(pi) - p._statePrev;

				if (l < _layers.size() - 1) {
					for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
						p._feedBackConnections[ci]._weight += _layerDescs[l]._learnFeedBack * predictionError * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._stateOutputPrev;
				}

				// Predictive
				for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
					p._predictiveConnections[ci]._weight += _layerDescs[l]._learnPrediction * predictionError * _layers[l]._sdr.getHiddenStatePrev(p._predictiveConnections[ci]._index);
			}

			float activation = 0.0f;

			// Feed Back
			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
					activation += p._feedBackConnections[ci]._weight * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._stateOutput;
			}

			// Predictive
			for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
				activation += p._predictiveConnections[ci]._weight * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);

			p._state = sigmoid(activation) * 2.0f - 1.0f;

			p._stateOutput = p._state;
		}
	}

	// Get first layer prediction
	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		int inputIndex = 0;

		for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
			p._sdrrl.setState(inputIndex++, _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._state);

		p._sdrrl.simStep(reward, _cellSparsity,_gamma,
			_gateSolveIter, _gateFeedForwardAlpha, _gateThresholdAlpha,
			_qAlpha, _actionAlpha, _actionDeriveIterations, _actionDeriveAlpha,
			_gammaLambda, _explorationStdDev, _explorationBreak,
			_averageSurpriseDecay, _surpriseLearnFactor, generator);

		// Learn
		if (learn) {
			float predictionError = p._sdrrl.getAction(0) * (p._stateOutputPrev - p._statePrev);

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::R))
				std::cout << p._sdrrl.getAction(0) << " ";

			for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
				p._feedBackConnections[ci]._weight += _learnFeedBack * predictionError * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._stateOutputPrev;// _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._statePrev;
		}

		float activation = 0.0f;

		// Feed Back
		for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
			activation += p._feedBackConnections[ci]._weight * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._stateOutput; //_layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._state;

		p._state = sigmoid(activation) * 2.0f - 1.0f;

		p._stateOutput = p._state;
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _explorationStdDev);

	// Add some noise and set as input by default
	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		if (dist01(generator) < _explorationBreak)
			p._stateOutput = dist01(generator) * 2.0f - 1.0f;
		else
			p._stateOutput = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, p._stateOutput)) + pertDist(generator)));
	}

	for (int l = 0; l < _layers.size(); l++) {
		if (learn)
			_layers[l]._sdr.learn(predictionErrors[l], _layerDescs[l]._sdrLambda, _layerDescs[l]._sdrBaselineDecay, _layerDescs[l]._sdrSensitivity, _layerDescs[l]._learnFeedForward, _layerDescs[l]._learnRecurrent, _layerDescs[l]._sdrLearnBoost, _layerDescs[l]._sdrBoostSparsity, _layerDescs[l]._sdrWeightDecay); //attentions[l], 

		_layers[l]._sdr.stepEnd();

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._statePrev = p._state;
			p._stateOutputPrev = p._stateOutput;
		}
	}

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		p._statePrev = p._state;
		p._stateOutputPrev = p._stateOutput;

		_layers.front()._sdr.setVisibleState(pi, p._stateOutput);
	}
}