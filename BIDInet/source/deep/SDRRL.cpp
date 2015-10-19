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

		_cells[i]._actionConnections.resize(numActions);

		for (int j = 0; j < numActions; j++)
			_cells[i]._actionConnections[j]._weight = weightDist(generator);

		_qConnections[i]._weight = weightDist(generator);
	}

	_actions.resize(numActions);
}

void SDRRL::simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float actionDeriveStdDev, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev); 
	std::normal_distribution<float> actionDeriveDist(0.0f, actionDeriveStdDev);

	for (int i = 0; i < _cells.size(); i++) {
		float activation = 0.0f;

		for (int j = 0; j < _inputs.size(); j++)
			activation += _cells[i]._feedForwardConnections[j]._weight * _inputs[j];

		_cells[i]._activation = activation;
	}

	// Inhibit
	for (int i = 0; i < _cells.size(); i++) {
		float inhibition = _cells[i]._bias._weight;

		for (int j = 0; j < _cells.size(); j++)
			if (_cells[i]._activation > _cells[j]._activation)
				inhibition += _cells[i]._lateralConnections[j]._weight;

		_cells[i]._state = _cells[i]._activation > inhibition ? 1.0f : 0.0f;
	}

	// Derive action
	float maxQ = 0.0f;

	// Init starting action randomly
	for (int i = 0; i < _actions.size(); i++) {
		_actions[i]._deriveState = dist01(generator) * 2.0f - 1.0f;
	}

	for (int iter = 0; iter < actionDeriveIterations; iter++) {
		maxQ = 0.0f;

		for (int i = 0; i < _cells.size(); i++) {
			float sum = 0.0f;

			for (int j = 0; j < _actions.size(); j++)
				sum += _cells[i]._actionConnections[j]._weight * _actions[j]._deriveState;

			_cells[i]._actionState = sigmoid(sum) * _cells[i]._state;

			maxQ += _qConnections[i]._weight * _cells[i]._actionState;
		}

		// Modify action to maximize Q
		for (int i = 0; i < _cells.size(); i++)
			_cells[i]._actionError = _qConnections[i]._weight * _cells[i]._actionState * (1.0f - _cells[i]._actionState);

		for (int i = 0; i < _actions.size(); i++) {
			float sum = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				sum += _cells[j]._actionConnections[i]._weight * _cells[j]._actionError;

			_actions[i]._deriveState = std::min(1.0f, std::max(-1.0f, _actions[i]._deriveState + actionDeriveAlpha * sum + actionDeriveDist(generator)));
		}

		//std::cout <<"MQ: " << maxQ << std::endl;
	}

	// Exploration
	for (int i = 0; i < _actions.size(); i++) {
		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator) * 2.0f - 1.0f;
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(-1.0f, _actions[i]._deriveState + pertDist(generator)));
	}

	float q = 0.0f;

	for (int i = 0; i < _cells.size(); i++) {
		float sum = 0.0f;

		for (int j = 0; j < _actions.size(); j++)
			sum += _cells[i]._actionConnections[j]._weight * _actions[j]._exploratoryState;

		_cells[i]._actionState = sigmoid(sum) * _cells[i]._state;

		q += _qConnections[i]._weight * _cells[i]._actionState;
	}

	float tdError = reward + gamma * maxQ - _prevValue;
	float qAlphaTdError = qAlpha * tdError;
	float actionAlphaTdError = actionAlpha * tdError;
	float surprise = tdError * tdError;

	float learnPattern = sigmoid(surpriseLearnFactor * (surprise - _averageSurprise));
	//std::cout << "LP: " << learnPattern << std::endl;
	_averageSurprise = (1.0f - averageSurpiseDecay) * _averageSurprise + averageSurpiseDecay * surprise;

	for (int i = 0; i < _cells.size(); i++) {
		float error = _qConnections[i]._weight * _cells[i]._actionState * (1.0f - _cells[i]._actionState);

		_qConnections[i]._weight += qAlphaTdError * _qConnections[i]._trace;
		_qConnections[i]._trace = gammaLambda * _qConnections[i]._trace + _cells[i]._actionState;

		for (int j = 0; j < _actions.size(); j++) {
			_cells[i]._actionConnections[j]._weight += actionAlphaTdError * _cells[i]._actionConnections[j]._trace;
			_cells[i]._actionConnections[j]._trace = gammaLambda * _cells[i]._actionConnections[j]._trace + error * _actions[j]._exploratoryState;
		}
	}

	// Reconstruct
	for (int i = 0; i < _reconstructionError.size(); i++) {
		float recon = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			recon += _cells[j]._feedForwardConnections[i]._weight * _cells[j]._state;

		_reconstructionError[i] = gateFeedForwardAlpha * learnPattern * (_inputs[i] - recon);
	}

	float sparsitySquared = sparsity * sparsity;

	for (int i = 0; i < _cells.size(); i++) {
		// Learn SDRs
		if (_cells[i]._state > 0.0f) {
			for (int j = 0; j < _inputs.size(); j++)
				_cells[i]._feedForwardConnections[j]._weight += _reconstructionError[j];
		}

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._lateralConnections[j]._weight = std::max(0.0f, _cells[i]._lateralConnections[j]._weight + gateLateralAlpha * learnPattern * (_cells[i]._state * _cells[j]._state - sparsitySquared)); //(_cells[i]._stateActivation > _cells[j]._stateActivation ? 1.0f : 0.0f)

		_cells[i]._bias._weight += gateBiasAlpha * (_cells[i]._state - sparsity);
	}

	_prevValue = q;
}