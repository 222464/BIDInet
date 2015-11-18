#include "Agent.h"

#include <algorithm>

using namespace neo;

void Agent::createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int widthPrev = inputWidth;
	int heightPrev = inputHeight;

	for (int l = 0; l < _layerDescs.size(); l++) {
		_layers[l]._sdr.createRandom(widthPrev, heightPrev, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._receptiveRadius, _layerDescs[l]._recurrentRadius, _layerDescs[l]._lateralRadius, initMinWeight, initMaxWeight, initMinInhibition, initMaxInhibition, initThreshold, generator);

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

			p._column.createRandom(p._predictiveConnections.size() + p._feedBackConnections.size() * 2, _numColumnActions, _layerDescs[l]._cellsPerColumn, initMinWeight, initMaxWeight, initMinInhibition, initMaxInhibition, initThreshold, generator);
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

		p._column.createRandom(p._feedBackConnections.size() * 2, _numColumnActions, _cellsPerColumn, initMinWeight, initMaxWeight, initMinInhibition, initMaxInhibition, initThreshold, generator);
	}
}

void Agent::simStep(float reward, std::mt19937 &generator, bool learn) {
	// Feature extraction
	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._sdr.activate(_layerDescs[l]._sdrIter, _layerDescs[l]._sdrLeak, generator);

		// Set inputs for next layer if there is one
		if (l < _layers.size() - 1) {
			for (int i = 0; i < _layers[l]._sdr.getNumHidden(); i++) {
				_layers[l + 1]._sdr.setVisibleState(i, _layers[l]._sdr.getHiddenState(i) * _layers[l]._predictionNodes[i]._column.getAction(_attention));
			}
		}
	}

	// Prediction
	for (int l = _layers.size() - 1; l >= 0; l--) {
		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			int colInputIndex = 0;

			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
					p._column.setState(colInputIndex++, _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._column.getAction(_signal));
					p._column.setState(colInputIndex++, _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._state);
				}
			}

			for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
				p._column.setState(colInputIndex++, _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index));

			// Update column
			p._column.simStep(reward, _layerDescs[l]._columnSparsity, _layerDescs[l]._columnGamma,
				_layerDescs[l]._columnIter, _layerDescs[l]._columnLeak,
				_layerDescs[l]._columnFeedForwardAlpha, _layerDescs[l]._columnLateralAlpha, _layerDescs[l]._columnThresholdAlpha,
				_layerDescs[l]._columnQAlpha, _layerDescs[l]._columnActionAlpha,
				_layerDescs[l]._columnGammaLambda,
				_layerDescs[l]._columnExplorationStdDev, _layerDescs[l]._columnExplorationBreakChance, generator);

			// Learn
			if (learn) {
				float predictionError = p._column.getAction(_learnPrediction) * (_layers[l]._sdr.getHiddenState(pi) - p._statePrev);

				if (l < _layers.size() - 1) {
					for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
						p._feedBackConnections[ci]._weight += _layerDescs[l]._learnFeedBack * predictionError * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._statePrev;
				}

				// Predictive
				for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
					p._predictiveConnections[ci]._weight += _layerDescs[l]._learnPrediction * predictionError * _layers[l]._sdr.getHiddenStatePrev(p._predictiveConnections[ci]._index);
			}

			float activation = 0.0f;

			// Feed Back
			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
					activation += p._feedBackConnections[ci]._weight * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._state;
			}

			// Predictive
			for (int ci = 0; ci < p._predictiveConnections.size(); ci++)
				activation += p._predictiveConnections[ci]._weight * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);

			p._activation = activation;

			p._state = std::min(1.0f, std::max(0.0f, p._activation));
		}
	}

	// Get first layer prediction
	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		int colInputIndex = 0;

		for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
			p._column.setState(colInputIndex++, _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._column.getAction(_signal));
			p._column.setState(colInputIndex++, _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._state);
		}

		// Update column
		p._column.simStep(reward, _columnSparsity, _columnGamma,
			_columnIter, _columnLeak,
			_columnFeedForwardAlpha, _columnLateralAlpha, _columnThresholdAlpha,
			_columnQAlpha, _columnActionAlpha,
			_columnGammaLambda,
			_columnExplorationStdDev, _columnExplorationBreakChance, generator);

		// Learn
		if (learn) {
			float predictionError = p._column.getAction(_learnPrediction) * (_layers.front()._sdr.getVisibleState(pi) - p._statePrev);

			for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
				p._feedBackConnections[ci]._weight += _learnInputFeedBack * predictionError * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._statePrev;
		}

		float activation = 0.0f;

		// Feed Back
		for (int ci = 0; ci < p._feedBackConnections.size(); ci++)
			activation += p._feedBackConnections[ci]._weight * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._state;

		p._activation = activation;

		p._state = p._activation;
	}

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<float> rewards(_layers[l]._predictionNodes.size());

		if (learn) {
			for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
				PredictionNode &p = _layers[l]._predictionNodes[pi];

				float predictionError = _layers[l]._sdr.getHiddenState(pi) - p._statePrev;

				float error2 = predictionError * predictionError;

				rewards[pi] = sigmoid(_layerDescs[l]._sdrSensitivity * (error2 - p._baseline));

				p._baseline = (1.0f - _layerDescs[l]._sdrBaselineDecay) * p._baseline + _layerDescs[l]._sdrBaselineDecay * error2;
			}

			_layers[l]._sdr.learn(rewards, _layerDescs[l]._sdrLambda, _layerDescs[l]._learnFeedForward, _layerDescs[l]._learnRecurrent, _layerDescs[l]._learnLateral, _layerDescs[l]._sdrLearnThreshold, _layerDescs[l]._sdrSparsity, _layerDescs[l]._sdrWeightDecay, _layerDescs[l]._sdrMaxWeightDelta); //attentions[l], 
		}

		_layers[l]._sdr.stepEnd();

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._statePrev = p._state;
			p._activationPrev = p._activation;
		}
	}

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputPredictionNode &p = _inputPredictionNodes[pi];

		p._statePrev = p._state;
		p._activationPrev = p._activation;
	}
}