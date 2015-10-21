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

	for (int i = 0; i < numCells; i++) {
		_cells[i]._feedForwardConnections.resize(_inputs.size());

		_cells[i]._bias._weight = weightDist(generator);

		_cells[i]._actionBias._weight = weightDist(generator);

		for (int j = 0; j < _inputs.size(); j++)
			_cells[i]._feedForwardConnections[j]._weight = weightDist(generator);

		_cells[i]._lateralConnections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._lateralConnections[j]._weight = inhibitionDist(generator);

		_cells[i]._actionConnections.resize(numActions);

		for (int j = 0; j < numActions; j++)
			_cells[i]._actionConnections[j]._weight = weightDist(generator);
	}

	_actions.resize(numActions);

	for (int i = 0; i < numActions; i++)
		_actions[i]._bias._weight = weightDist(generator);
}

void SDRRL::simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float actionDeriveStdDev, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator) {
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
	float zInv = 0.0f;

	for (int i = 0; i < _cells.size(); i++) {
		float inhibition = _cells[i]._bias._weight;

		for (int j = 0; j < _cells.size(); j++)
			if (_cells[i]._activation > _cells[j]._activation)
				inhibition += _cells[i]._lateralConnections[j]._weight;

		_cells[i]._state = 1.0f > inhibition ? 1.0f : 0.0f;

		zInv += _cells[i]._state;
	}

	zInv = 1.0f / std::sqrt(zInv + _actions.size());

	// Init starting action randomly
	for (int i = 0; i < _actions.size(); i++) {
		_actions[i]._deriveState = dist01(generator) * 2.0f - 1.0f;
	}

	for (int iter = 0; iter < actionDeriveIterations; iter++) {
		for (int k = 0; k < _cells .size(); k++) {
			if (_cells[k]._state > 0.0f){
				float sum = _cells[k]._actionBias._weight;

				for (int vi = 0; vi < _actions.size(); vi++)
					sum += _cells[k]._actionConnections[vi]._weight * _actions[vi]._deriveState;

				_cells[k]._actionState = sigmoid(sum) * _cells[k]._state;
			}
			else
				_cells[k]._actionState = 0.0f;
		}

		// Modify action to maximize Q
		for (int j = 0; j < _actions.size(); j++) {
			float sum = _actions[j]._bias._weight;

			for (int k = 0; k < _cells.size(); k++)
				sum += _cells[k]._actionConnections[j]._weight * _cells[k]._actionState;

			_actions[j]._deriveState = std::min(1.0f, std::max(-1.0f, _actions[j]._deriveState + actionDeriveAlpha * sum));// + actionDeriveDist(generator)
		}

		//std::cout <<"MQ: " << maxQ << std::endl;
	}
	
	float maxQ = 0.0f;

	{
		float freeEnergy = 0.0f;

		for (int k = 0; k < _cells.size(); k++) {
			freeEnergy -= _cells[k]._actionBias._weight * _cells[k]._actionState;

			for (int vi = 0; vi < _actions.size(); vi++)
				freeEnergy -= _cells[k]._actionConnections[vi]._weight * _actions[vi]._deriveState * _cells[k]._actionState;

			//sum += _hidden[k]._state * std::log(_hidden[k]._state) + (1.0f - _hidden[k]._state) * std::log(1.0f - _hidden[k]._state);
		}

		for (int vi = 0; vi < _actions.size(); vi++)
			freeEnergy -= _actions[vi]._bias._weight * _actions[vi]._deriveState;

		maxQ = -freeEnergy * zInv;
	}

	// Exploration
	for (int i = 0; i < _actions.size(); i++) {
		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator) * 2.0f - 1.0f;
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(-1.0f, _actions[i]._deriveState + pertDist(generator)));
	}

	float q = 0.0f;

	for (int k = 0; k < _cells.size(); k++) {
		if (_cells[k]._state > 0.0f){
			float sum = _cells[k]._actionBias._weight;

			for (int vi = 0; vi < _actions.size(); vi++)
				sum += _cells[k]._actionConnections[vi]._weight * _actions[vi]._exploratoryState;

			_cells[k]._actionState = sigmoid(sum) * _cells[k]._state;
		}
		else
			_cells[k]._actionState = 0.0f;
	}

	{
		float freeEnergy = 0.0f;

		for (int k = 0; k < _cells.size(); k++) {
			freeEnergy -= _cells[k]._actionBias._weight * _cells[k]._actionState;

			for (int vi = 0; vi < _actions.size(); vi++)
				freeEnergy -= _cells[k]._actionConnections[vi]._weight * _actions[vi]._exploratoryState * _cells[k]._actionState;

			//sum += _hidden[k]._state * std::log(_hidden[k]._state) + (1.0f - _hidden[k]._state) * std::log(1.0f - _hidden[k]._state);
		}

		for (int vi = 0; vi < _actions.size(); vi++)
			freeEnergy -= _actions[vi]._bias._weight * _actions[vi]._exploratoryState;

		q = -freeEnergy * zInv;
	}

	float tdError = reward + gamma * q - _prevValue;
	float actionAlphaTdError = actionAlpha * tdError;
	float surprise = tdError * tdError;

	float learnPattern = sigmoid(surpriseLearnFactor * (surprise - _averageSurprise));
	//std::cout << "LP: " << learnPattern << std::endl;
	_averageSurprise = (1.0f - averageSurpiseDecay) * _averageSurprise + averageSurpiseDecay * surprise;

	// Update weights
	for (int k = 0; k < _cells.size(); k++) {
		_cells[k]._actionBias._weight += actionAlphaTdError * _cells[k]._actionBias._trace;

		_cells[k]._actionBias._trace = _cells[k]._actionBias._trace * gammaLambda + _cells[k]._actionState;

		for (int vi = 0; vi < _actions.size(); vi++) {
			_cells[k]._actionConnections[vi]._weight += actionAlphaTdError * _cells[k]._actionConnections[vi]._trace;

			_cells[k]._actionConnections[vi]._trace = _cells[k]._actionConnections[vi]._trace * gammaLambda + _cells[k]._actionState * _actions[vi]._exploratoryState;
		}
	}

	for (int vi = 0; vi < _actions.size(); vi++) {
		_actions[vi]._bias._weight += actionAlphaTdError * _actions[vi]._bias._trace;

		_actions[vi]._bias._trace = _actions[vi]._bias._trace * gammaLambda + _actions[vi]._exploratoryState;
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