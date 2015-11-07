#include "MiniQ.h"

#include <algorithm>

using namespace deep;

void MiniQ::createRandom(int numStates, int numQ, float initMinWeight, float initMaxWeight, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_states.clear();
	_states.assign(numStates, 0.0f);

	_qNodes.resize(numQ);

	for (int i = 0; i < _qNodes.size(); i++) {
		_qNodes[i]._connections.resize(_states.size());

		for (int j = 0; j < _states.size(); j++)
			_qNodes[i]._connections[j]._weight = weightDist(generator);
	}
}

int MiniQ::simStep(float reward, float alpha, float gamma, float lambdaGamma, float epsilon, std::mt19937 &generator) {
	// Compute Q values
	int maxQIndex = 0;

	float maxQ = 0.0f;

	for (int j = 0; j < _states.size(); j++)
		maxQ += _qNodes.front()._connections[j]._weight * _states[j];

	for (int i = 1; i < _qNodes.size(); i++) {
		float q = 0.0f;

		for (int j = 0; j < _states.size(); j++)
			q += _qNodes[i]._connections[j]._weight * _states[j];

		if (q > maxQ) {
			maxQ = q;

			maxQIndex = i;
		}
	}

	// Select action
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::uniform_int_distribution<int> actionDist(0, _qNodes.size() - 1);

	int action = maxQIndex;

	if (dist01(generator) < epsilon)
		action = actionDist(generator);

	float tdError = reward + gamma * maxQ - _prevValue;

	_prevValue = maxQ;

	// Weight and trace update
	for (int i = 0; i < _qNodes.size(); i++) {
		for (int j = 0; j < _states.size(); j++) {
			_qNodes[i]._connections[j]._weight += alpha * tdError * _qNodes[i]._connections[j]._trace;

			_qNodes[i]._connections[j]._trace = std::max(lambdaGamma * _qNodes[i]._connections[j]._trace, i == action ? _states[j] : 0.0f);
		}
	}

	return action;
}