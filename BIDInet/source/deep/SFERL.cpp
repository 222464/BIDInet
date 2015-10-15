#include "SFERL.h"

#include <algorithm>

#include <iostream>

using namespace deep;

SFERL::SFERL()
	: _zInv(1.0f), _prevValue(0.0f)
{}

void SFERL::createRandom(int numState, int numAction, int numHidden, float weightStdDev, float maxInhibition, std::mt19937 &generator) {
	_numState = numState;
	_numAction = numAction;

	_visible.resize(numState + numAction);

	_hidden.resize(numHidden);

	_actions.resize(numAction);

	std::normal_distribution<float> weightDist(0.0f, weightStdDev);
	std::uniform_real_distribution<float> inhibitionDist(0.0f, maxInhibition);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight = weightDist(generator);

	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight = weightDist(generator);

		_hidden[k]._feedForwardConnections.resize(_visible.size());

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._feedForwardConnections[vi]._weight = weightDist(generator);

		_hidden[k]._lateralConnections.resize(_hidden.size());

		for (int ko = 0; ko < _hidden.size(); ko++)
			_hidden[k]._lateralConnections[ko] = inhibitionDist(generator);
	}

	for (int a = 0; a < _actions.size(); a++) {
		_actions[a]._bias._weight = weightDist(generator);

		_actions[a]._feedForwardConnections.resize(_hidden.size());

		for (int vi = 0; vi < _hidden.size(); vi++)
			_actions[a]._feedForwardConnections[vi]._weight = weightDist(generator);
	}

	_zInv = 1.0f / std::sqrt(static_cast<float>(numState + numHidden));

	_prevVisible.clear();
	_prevVisible.assign(_visible.size(), 0.0f);

	_prevHidden.clear();
	_prevHidden.assign(_hidden.size(), 0.0f);
}

float SFERL::freeEnergy() const {
	float sum = 0.0f;

	for (int k = 0; k < _hidden.size(); k++) {
		sum -= _hidden[k]._bias._weight * _hidden[k]._state;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum -= _hidden[k]._feedForwardConnections[vi]._weight * _visible[vi]._state * _hidden[k]._state;

		//sum += _hidden[k]._state * std::log(_hidden[k]._state) + (1.0f - _hidden[k]._state) * std::log(1.0f - _hidden[k]._state);
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		sum -= _visible[vi]._bias._weight * _visible[vi]._state;

	return sum;
}

void SFERL::step(const std::vector<float> &state, std::vector<float> &action,
	float reward, float gamma, float lambdaGamma, float actionSearchAlpha,
	float lateralAlpha, float actionAlpha, float gradientAlpha, float sparsitySquared,
	float breakChance, float perturbationStdDev,
	std::mt19937 &generator)
{
	for (int i = 0; i < _numState; i++)
		_visible[i]._state = state[i];

	activate();

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	// Actual action (perturbed from maximum)
	if (action.size() != _numAction)
		action.resize(_numAction);

	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	// Find last best action
	for (int a = 0; a < _actions.size(); a++) {
		float sum = _actions[a]._bias._weight;

		for (int k = 0; k < _hidden.size(); k++)
			sum += _actions[a]._feedForwardConnections[k]._weight * _hidden[k]._state;

		_actions[a]._state = sum;

		int index = a + _numState;

		float error = _visible[index]._bias._weight;

		for (int k = 0; k < _hidden.size(); k++)
			error += _hidden[k]._feedForwardConnections[index]._weight * _hidden[k]._state;

		_visible[a + _numState]._state = std::min(1.0f, std::max(-1.0f, sum + actionSearchAlpha * error));

		if (uniformDist(generator) < breakChance)
			action[a] = uniformDist(generator) * 2.0f - 1.0f;
		else
			action[a] = std::min(1.0f, std::max(-1.0f, _visible[a + _numState]._state + perturbationDist(generator)));
	}

	float predictedQ = value();

	// Update Q
	float tdError = reward + gamma * predictedQ - _prevValue;

	updateOnError(gradientAlpha * tdError, lateralAlpha, sparsitySquared, lambdaGamma);

	// Update
	for (int a = 0; a < _actions.size(); a++) {
		float alphaError = actionAlpha * (_visible[a + _numState]._state - _actions[a]._state);

		_actions[a]._bias._weight += alphaError;

		for (int k = 0; k < _hidden.size(); k++)
			_actions[a]._feedForwardConnections[k]._weight += alphaError * _hidden[k]._state;
	}

	_prevValue = predictedQ;

	for (int j = 0; j < _numAction; j++)
		_visible[_numState + j]._state = action[j];

	for (int i = 0; i < _visible.size(); i++)
		_prevVisible[i] = _visible[i]._state;

	for (int i = 0; i < _hidden.size(); i++)
		_prevHidden[i] = _hidden[i]._state;
}

void SFERL::activate() {
	for (int k = 0; k < _hidden.size(); k++) {
		float sum = _hidden[k]._bias._weight;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum += _hidden[k]._feedForwardConnections[vi]._weight * _visible[vi]._state;

		_hidden[k]._activation = sum;
	}

	// Sparsify
	for (int k = 0; k < _hidden.size(); k++) {
		float inhibition = 0.0f;

		for (int ko = 0; ko < _hidden.size(); ko++)
			inhibition += _hidden[k]._activation > _hidden[ko]._activation ? _hidden[k]._lateralConnections[ko] : 0.0f;

		if (inhibition < 1.0f)
			_hidden[k]._state = sigmoid(_hidden[k]._activation);
		else
			_hidden[k]._state = 0.0f;
	}
}

void SFERL::updateOnError(float error, float lateralAlpha, float sparsitySquared, float lambdaGamma) {
	// Update inhibition
	for (int k = 0; k < _hidden.size(); k++) {
		float sState = (_hidden[k]._state > 0.0f ? 1.0f : 0.0f);

		for (int ko = 0; ko < _hidden.size(); ko++)
			_hidden[k]._lateralConnections[ko] = std::max(0.0f, _hidden[k]._lateralConnections[ko] + lateralAlpha * (sState * (_hidden[ko]._state > 0.0f ? 1.0f : 0.0f) - sparsitySquared));
	}

	// Update weights
	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight += error * _hidden[k]._bias._trace;

		_hidden[k]._bias._trace = lambdaGamma * _hidden[k]._bias._trace + _hidden[k]._state;

		for (int vi = 0; vi < _visible.size(); vi++) {
			_hidden[k]._feedForwardConnections[vi]._weight += error * _hidden[k]._feedForwardConnections[vi]._trace;

			_hidden[k]._feedForwardConnections[vi]._trace = lambdaGamma * _hidden[k]._feedForwardConnections[vi]._trace + _hidden[k]._state * _visible[vi]._state;
		}
	}

	for (int vi = 0; vi < _visible.size(); vi++) {
		_visible[vi]._bias._weight += error * _visible[vi]._bias._trace;

		_visible[vi]._bias._trace = lambdaGamma * _visible[vi]._bias._trace + _visible[vi]._state;
	}
}