#include "CSRL.h"

#include <algorithm>

#include <SFML/Window.hpp>

#include <iostream>

using namespace deep;

void CSRL::createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	_layerDescs = layerDescs;

	_inputTypes = inputTypes;

	for (int i = 0; i < _inputTypes.size(); i++) {
		if (_inputTypes[i] == _q) {
			QNode n;

			n._index = i;

			n._offset = dist01(generator) * 2.0f - 1.0f;

			_qNodes.push_back(n);
		}
	}

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
	}
}

void CSRL::simStep(float reward, std::mt19937 &generator, bool learn) {
	// Feature extraction
	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._sdr.activate(_layerDescs[l]._sdrIterSettle, _layerDescs[l]._sdrStepSize, _layerDescs[l]._sdrHiddenDecay, _layerDescs[l]._sdrNoise, generator);

		// Set inputs for next layer if there is one
		if (l < _layers.size() - 1) {
			for (int i = 0; i < _layers[l]._sdr.getNumHidden(); i++) {
				// Attention gate
				float gated = _layers[l]._sdr.getHiddenState(i) * (_layers[l]._predictionNodes[i]._localReward > 0.0f ? 0.0f : 1.0f);

				_layers[l + 1]._sdr.setVisibleState(i, gated);
			}
		}
	}

	// Prediction
	for (int l = _layers.size() - 1; l >= 0; l--) {
		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];
		
			float activation = 0.0f;

			// Feed Back
			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
					activation += p._feedBackConnections[ci]._weight * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._stateOutput;
			}

			// Predictive
			for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
				activation += p._predictiveConnections[ci]._weight * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);

			p._state = activation;

			p._stateOutput = std::min(1.0f, std::max(-1.0f, p._state));
		}
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _explorationStdDev);

	// Get first layer prediction
	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		float activation = 0.0f;

		// Feed Back
		for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
			activation += p._feedBackConnections[ci]._weight * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._stateOutput; //_layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._state;

		p._state = activation;

		p._stateOutput = std::min(1.0f, std::max(-1.0f, p._state));

		// Add noise
		if (_inputTypes[pi] == _action) {
			if (dist01(generator) < _explorationBreak)
				p._stateOutput = dist01(generator) * 2.0f - 1.0f;
			else
				p._stateOutput = std::min(1.0f, std::max(-1.0f, p._stateOutput + pertDist(generator)));
		}
	}

	// Gather Q
	float q = 0.0f;

	for (int i = 0; i < _qNodes.size(); i++)
		q += _inputPredictionNodes[_qNodes[i]._index]._stateOutput - _qNodes[i]._offset;

	q /= _qNodes.size();

	// Compute TD error
	float tdError = reward + q * _gamma - _prevValue;

	float newQ = _prevValue + tdError * _qAlpha;
	std::cout << newQ << " " << tdError << std::endl;
	_prevValue = q;

	// Set local rewards for first layer to be td error
	for (int i = 0; i < _inputPredictionNodes.size(); i++)
		_inputPredictionNodes[i]._localReward = tdError;

	// Zero all other local rewards
	for (int l = 0; l < _layers.size(); l++) {
		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._localReward = 0.0f;
		}
	}

	// Propagate to first layer
	for (int i = 0; i < _inputPredictionNodes.size(); i++) {
		InputPredictionNode &p = _inputPredictionNodes[i];

		for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
			_layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._localReward += p._feedBackConnections[ci]._weight * p._localReward;
	}

	// Propagate all other local rewards backwards
	std::vector<std::vector<float>> rewards(_layers.size());

	for (int l = 0; l < _layers.size() - 1; l++) {
		rewards[l].resize(_layers[l]._predictionNodes.size());
		
		int nextLayerIndex = l + 1;

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			rewards[l][pi] = p._localReward > 0.0f ? 1.0f : 0.0f;

			// Propate my local reward to next layer
			for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
				_layers[nextLayerIndex]._predictionNodes[p._feedBackConnections[ci]._index]._localReward += p._feedBackConnections[ci]._weight * p._localReward;
		}
	}

	// Save reward of last layer as well
	for (int pi = 0; pi < _layers.back()._predictionNodes.size(); pi++) {
		rewards.back().resize(_layers.back()._predictionNodes.size());

		PredictionNode &p = _layers.back()._predictionNodes[pi];

		rewards.back()[pi] = p._localReward;
	}

	// Learning
	for (int l = 0; l < _layers.size(); l++) {	
		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			float predictionError = _layers[l]._sdr.getHiddenState(pi) - p._statePrev;

			float gate = p._localReward > 0.0f ? 1.0f : 0.0f;

			// Learn
			if (learn) {
				if (l < _layers.size() - 1) {
					for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
						p._feedBackConnections[ci]._trace = _layerDescs[l]._gammaLambda * p._feedBackConnections[ci]._trace + predictionError * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._stateOutputPrev;

						p._feedBackConnections[ci]._weight += _layerDescs[l]._learnFeedBack * gate * p._feedBackConnections[ci]._trace;
					}
				}

				// Predictive
				for (int ci = 0; ci < p._predictiveConnections.size(); ci++) {
					p._predictiveConnections[ci]._trace = _layerDescs[l]._gammaLambda * p._predictiveConnections[ci]._trace + predictionError * _layers[l]._sdr.getHiddenStatePrev(p._predictiveConnections[ci]._index);

					p._predictiveConnections[ci]._weight += _layerDescs[l]._learnPrediction * gate * p._predictiveConnections[ci]._trace;
				}
			}
		}
	}

	// Get first layer prediction
	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		float predictionError = _layers.front()._sdr.getVisibleState(pi) - p._statePrev;

		float gate = p._localReward > 0.0f ? 1.0f : 0.0f;

		// Learn
		if (learn) {
			for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
				p._feedBackConnections[ci]._trace = _gammaLambda * p._feedBackConnections[ci]._trace + predictionError * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._stateOutputPrev;

				p._feedBackConnections[ci]._weight += _learnFeedBack * gate * p._feedBackConnections[ci]._trace;
			}
		}
	}

	for (int l = 0; l < _layers.size(); l++) {
		if (learn)
			_layers[l]._sdr.learn(_layerDescs[l]._learnFeedForward, _layerDescs[l]._learnRecurrent, _layerDescs[l]._sdrLearnThreshold, _layerDescs[l]._sparsity, _layerDescs[l]._sdrWeightDecay); //attentions[l], 

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

	// Set Q
	for (int i = 0; i < _qNodes.size(); i++)
		_layers.front()._sdr.setVisibleState(_qNodes[i]._index, newQ + _qNodes[i]._offset);
}