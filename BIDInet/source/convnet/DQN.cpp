#include "DQN.h"

#include <algorithm>

#include <iostream>

using namespace convnet;

void DQN::simStep(float reward, std::mt19937 &generator) {
	if (_exploratoryActions.empty()) {
		_exploratoryActions.assign(_net.getNumOutputs() - 1, 0.0f);

		_actions.assign(_exploratoryActions.size(), 0.0f);

		_inputMapsPrev = _net.getLayer(0)->getOutputMaps();
	}

	std::vector<float> exploratoryActionsPrev = _exploratoryActions;
	std::vector<float> actionsPrev = _actions;

	// Actions and Q
	_net.forward();

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _actionPerturbationStdDev);

	for (int a = 0; a < _exploratoryActions.size(); a++) {
		_actions[a] = _net.getOutput(a);

		if (dist01(generator) < _actionBreakChance)
			_exploratoryActions[a] = dist01(generator) * 2.0f - 1.0f;
		else
			_exploratoryActions[a] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _actions[a])) + pertDist(generator)));
	}

	float q = _net.getOutput(_net.getNumOutputs() - 1);

	float tdError = reward + _qGamma * q - _prevValue;

	float newQ = _prevValue + _qAlpha * tdError;

	_prevValue = q;

	// Update previous samples
	float g = _qGamma;
	
	for (std::list<ReplaySample>::iterator it = _replaySamples.begin(); it != _replaySamples.end(); it++) {
		it->_q += _qAlpha * tdError * g;

		g *= _qGamma;
	}

	// Create replay sample
	ReplaySample sample;
	sample._inputs = _inputMapsPrev;
	sample._q = newQ;
	sample._originalQ = _prevValue;
	sample._actions = exploratoryActionsPrev;
	sample._originalActions = actionsPrev;

	_inputMapsPrev = _net.getLayer(0)->getOutputMaps();

	_replaySamples.push_front(sample);

	while (_replaySamples.size() > _maxReplayChainSize)
		_replaySamples.pop_back();

	// Create randomly accessible version of replay samples
	std::vector<ReplaySample*> randomAccessSamples(_replaySamples.size());

	int index = 0;

	for (std::list<ReplaySample>::iterator it = _replaySamples.begin(); it != _replaySamples.end(); it++)
		randomAccessSamples[index++] = &(*it);
	
	std::uniform_int_distribution<int> sampleDist(0, randomAccessSamples.size() - 1);

	for (int i = 0; i < _replayIterations; i++) {
		int replayIndex = sampleDist(generator);

		_net.getLayer(0)->getOutputMaps() = randomAccessSamples[replayIndex]->_inputs;

		_net.forward();

		_net.setError(_actions.size(), randomAccessSamples[replayIndex]->_q - _net.getOutput(_actions.size()));
		
		if (randomAccessSamples[replayIndex]->_q > randomAccessSamples[replayIndex]->_originalQ) {
			for (int j = 0; j < _actions.size(); j++)
				_net.setError(j, randomAccessSamples[replayIndex]->_actions[j] - _net.getOutput(j));
		}
		else {
			for (int j = 0; j < _actions.size(); j++)
				_net.setError(j, randomAccessSamples[replayIndex]->_originalActions[j] - _net.getOutput(j));
		}

		_net.backward();

		_net.update();
	}
}