#include "SDRRL.h"

#include <algorithm>

#include <iostream>

using namespace deep;

void SDRRL::createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	std::uniform_real_distribution<float> inhibitionDist(initMinInhibition, initMaxInhibition);

	_numStates = numStates;

	_inputs.assign(numStates, 0.0f);
	_reconstructionError.assign(_inputs.size(), 0.0f);

	_cells.resize(numCells);

	_qConnections.resize(_cells.size());

	for (int i = 0; i < numCells; i++) {
		_cells[i]._feedForwardConnections.resize(_inputs.size());

		_cells[i]._bias._weight = weightDist(generator);

		for (int j = 0; j < _inputs.size(); j++)
			_cells[i]._feedForwardConnections[j]._weight = weightDist(generator);

		_cells[i]._lateralConnections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._lateralConnections[j]._weight = inhibitionDist(generator);

		_qConnections[i]._weight = weightDist(generator);

		_cells[i]._tdConnection._weight = weightDist(generator);
	}

	_actions.resize(numActions);

	for (int i = 0; i < _actions.size(); i++) {
		_actions[i]._connections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_actions[i]._connections[j]._weight = weightDist(generator);
	}
}

void SDRRL::simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev); 

	for (int i = 0; i < _cells.size(); i++) {
		float activation = 0.0f;

		for (int j = 0; j < _inputs.size(); j++)
			activation += _cells[i]._feedForwardConnections[j]._weight * _inputs[j];

		_cells[i]._activation = activation;
	}

	float q = 0.0f;

	// Inhibit
	for (int i = 0; i < _cells.size(); i++) {
		float inhibition = _cells[i]._bias._weight;

		for (int j = 0; j < _cells.size(); j++)
			if (_cells[i]._activation > _cells[j]._activation)
				inhibition += _cells[i]._lateralConnections[j]._weight;

		_cells[i]._state = inhibition < 1.0f ? 1.0f : 0.0f;

		q += _qConnections[i]._weight * _cells[i]._state;
	}

	float tdError = reward + gamma * q - _prevValue;
	float qAlphaTdError = qAlpha * tdError;

	// Update previous action
	for (int i = 0; i < _cells.size(); i++)
		_cells[i]._actionState = sigmoid(_cells[i]._tdConnection._weight * tdError) * _cells[i]._statePrev;

	for (int i = 0; i < _actions.size(); i++) {
		float action = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			action += _actions[i]._connections[j]._weight * _cells[j]._actionState;

		_actions[i]._error = _actions[i]._exploratoryState - sigmoid(action);
	}

	for (int i = 0; i < _cells.size(); i++) {
		float sum = 0.0f;

		for (int j = 0; j < _actions.size(); j++)
			sum += _actions[j]._error * _actions[j]._connections[i]._weight;

		_cells[i]._actionError = sum * _cells[i]._actionState * (1.0f - _cells[i]._actionState);

		_cells[i]._tdConnection._weight += actionAlpha * _cells[i]._actionError * tdError;
	}

	for (int i = 0; i < _actions.size(); i++) {
		for (int j = 0; j < _cells.size(); j++)
			_actions[i]._connections[j]._weight += actionAlpha * _actions[i]._error * _cells[j]._actionState;
	}

	// Derive new action
	for (int i = 0; i < _actions.size(); i++) {
		float action = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			action += _actions[i]._connections[j]._weight * (_cells[j]._tdConnection._weight > 0.0f ? 1.0f : 0.0f) * _cells[j]._state;

		_actions[i]._state = sigmoid(action);

		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator);
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(-1.0f, _actions[i]._state + pertDist(generator)));
	}

	// Reconstruct
	for (int i = 0; i < _reconstructionError.size(); i++) {
		float recon = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			recon += _cells[j]._feedForwardConnections[i]._weight * _cells[j]._state;

		_reconstructionError[i] = gateFeedForwardAlpha * (_inputs[i] - recon);
	}

	float sparsitySquared = sparsity * sparsity;

	for (int i = 0; i < _cells.size(); i++) {
		// Learn SDRs
		if (_cells[i]._state > 0.0f) {
			for (int j = 0; j < _inputs.size(); j++)
				_cells[i]._feedForwardConnections[j]._weight += _reconstructionError[j];
		}

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._lateralConnections[j]._weight = std::max(0.0f, _cells[i]._lateralConnections[j]._weight + gateLateralAlpha * (_cells[i]._state * _cells[j]._state - sparsitySquared)); //(_cells[i]._stateActivation > _cells[j]._stateActivation ? 1.0f : 0.0f)

		_cells[i]._bias._weight += gateBiasAlpha * (_cells[i]._state - sparsity);

		// Learn Q
		_qConnections[i]._weight += qAlphaTdError * _cells[i]._trace;
	}

	// Buffer update
	for (int i = 0; i < _cells.size(); i++) {
		_cells[i]._trace = std::max(_cells[i]._trace * gammaLambda, _cells[i]._state);

		_cells[i]._statePrev = _cells[i]._state;
	}

	_prevValue = q;
}