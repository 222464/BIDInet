#include "SelfOptimizingUnit.h"

#include <algorithm>

#include <iostream>

using namespace deep;

void SelfOptimizingUnit::createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	std::uniform_real_distribution<float> inhibitionDist(initMinInhibition, initMaxInhibition);

	_inputs.assign(numStates + numActions, 0.0f);
	_reconstruction.assign(_inputs.size(), 0.0f);

	_cells.resize(numCells);

	_qConnections.resize(_cells.size());

	for (int i = 0; i < numCells; i++) {
		_cells[i]._gateFeedForwardConnections.resize(_inputs.size());
		_cells[i]._stateConnections.resize(_inputs.size());

		_cells[i]._gateBias._weight = weightDist(generator);
		_cells[i]._stateBias._weight = weightDist(generator);

		for (int j = 0; j < _inputs.size(); j++) {
			_cells[i]._gateFeedForwardConnections[j]._weight = weightDist(generator);
			_cells[i]._stateConnections[j]._weight = weightDist(generator);
		}

		_cells[i]._gateLateralConnections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._gateLateralConnections[j]._weight = inhibitionDist(generator);

		_qConnections[i]._weight = weightDist(generator);
	}

	_actions.resize(numActions);

	for (int i = 0; i < _actions.size(); i++) {
		_actions[i]._bias._weight = weightDist(generator);
		
		_actions[i]._connections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_actions[i]._connections[j]._weight = weightDist(generator);
	}
}

void SelfOptimizingUnit::simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float stateLinearAlpha, float stateNonlinearAlpha, float actionAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, std::mt19937 &generator) {
	int actionsStartIndex = getNumStates();
	
	// Activate on new state and previous action
	for (int i = 0; i < _actions.size(); i++)
		_inputs[i + actionsStartIndex] = _actions[i]._exploratoryState;

	for (int i = 0; i < _cells.size(); i++) {
		float activation = _cells[i]._gateBias._weight;

		for (int j = 0; j < _inputs.size(); j++) {
			float delta = _cells[i]._gateFeedForwardConnections[j]._weight - _inputs[j];
		
			activation += -delta * delta;
		}

		_cells[i]._gateActivation = activation;
	}

	float q = 0.0f;

	// Inhibit
	for (int i = 0; i < _cells.size(); i++) {
		float inhibition = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			inhibition += _cells[i]._gateActivation < _cells[j]._gateActivation ? _cells[i]._gateLateralConnections[j]._weight : 0.0f;

		if (inhibition < 1.0f) {
			_cells[i]._gate = 1.0f;

			// Find state
			float activation = _cells[i]._stateBias._weight;

			for (int j = 0; j < _inputs.size(); j++)
				activation += _cells[i]._stateConnections[j]._weight * _inputs[j];

			_cells[i]._state = sigmoid(activation);
		}
		else {
			_cells[i]._gate = 0.0f;

			_cells[i]._state = 0.0f;
		}

		q += _qConnections[i]._weight * _cells[i]._state;
	}

	float tdError = reward + gamma * q - _prevValue;
	float stateLinearAlphaTdError = stateLinearAlpha * tdError;
	float stateNonlinearAlphaTdError = stateNonlinearAlpha * tdError;

	_prevValue = q;

	// Reconstruct
	for (int i = 0; i < _reconstruction.size(); i++) {
		float recon = 0.0f;
		float div = 0.0f;

		for (int j = 0; j < _cells.size(); j++) {
			recon += _cells[j]._gateFeedForwardConnections[i]._weight * _cells[j]._gate;

			div += _cells[j]._gate;
		}

		_reconstruction[i] = recon / std::max(1.0f, div);
	}

	float sparsitySquared = sparsity * sparsity;

	for (int i = 0; i < _cells.size(); i++) {
		// Learn SDRs
		if (_cells[i]._gate > 0.0f) {
			for (int j = 0; j < _inputs.size(); j++)
				_cells[i]._gateFeedForwardConnections[j]._weight += gateFeedForwardAlpha * (_inputs[j] - _reconstruction[j]);
		}

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._gateLateralConnections[j]._weight = std::max(0.0f, _cells[i]._gateLateralConnections[j]._weight + gateLateralAlpha * (_cells[i]._gate * (_cells[i]._gateActivation > _cells[j]._gateActivation ? 1.0f : 0.0f) - sparsitySquared));

		_cells[i]._gateBias._weight += gateBiasAlpha * (sparsity - _cells[i]._gate);

		// Learn states
		float stateError = _qConnections[i]._weight * _cells[i]._state * (1.0f - _cells[i]._state);

		for (int j = 0; j < _inputs.size(); j++) {
			_cells[i]._stateConnections[j]._weight += stateNonlinearAlphaTdError * _cells[i]._stateConnections[j]._trace;
		
			_cells[i]._stateConnections[j]._trace = _cells[i]._stateConnections[j]._trace * gammaLambda + stateError * _inputs[j];
		}

		_cells[i]._stateBias._weight += stateNonlinearAlphaTdError * _cells[i]._stateBias._trace;

		_cells[i]._stateBias._trace = _cells[i]._stateBias._trace * gammaLambda + stateError;

		// Learn Q
		_qConnections[i]._weight += stateLinearAlphaTdError * _qConnections[i]._trace;

		_qConnections[i]._trace = _qConnections[i]._trace * gammaLambda + _cells[i]._state;

		// Error
		_cells[i]._error = _qConnections[i]._weight * _cells[i]._state * (1.0f - _cells[i]._state);
	}

	// Optimize actions
	float actionAlphaTdError = actionAlpha * (tdError > 0.0f ? 1.0f : 0.0f);

	//std::cout << tdError << " " << q << std::endl;

	for (int i = 0; i < _actions.size(); i++) {
		float error = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			error += _cells[j]._stateConnections[actionsStartIndex + i]._weight * _cells[j]._error;

		float delta = ((error > 0.0f ? 1.0f : -1.0f) + _actions[i]._exploratoryState - _actions[i]._state) * _actions[i]._state * (1.0f - _actions[i]._state);

		// Update actions base on previous state
		for (int j = 0; j < _cells.size(); j++) {		
			_actions[i]._connections[j]._trace = _actions[i]._connections[j]._trace * gammaLambda + delta * _cells[j]._statePrev;

			// Trace order update reverse here on purpose since action is based on previous state
			_actions[i]._connections[j]._weight += actionAlphaTdError * _actions[i]._connections[j]._trace;

			//_actions[i]._connections[j]._weight += actionAlpha * delta * _cells[j]._statePrev;
		}

		_actions[i]._bias._trace = _actions[i]._bias._trace * gammaLambda + error;

		// Trace order update reverse here on purpose since action is based on previous state
		_actions[i]._bias._weight += stateNonlinearAlphaTdError * _actions[i]._bias._trace;
	}

	// Find new actions
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev);

	for (int i = 0; i < _actions.size(); i++) {
		float activation = _actions[i]._bias._weight;

		for (int j = 0; j < _cells.size(); j++)
			activation += _actions[i]._connections[j]._weight * _cells[j]._state;

		_actions[i]._state = sigmoid(activation);

		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator);
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(0.0f, _actions[i]._state + pertDist(generator)));
	}

	// Buffer update
	for (int i = 0; i < _cells.size(); i++)
		_cells[i]._statePrev = _cells[i]._state;
}