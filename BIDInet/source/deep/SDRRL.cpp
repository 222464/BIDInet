#include "SDRRL.h"

#include <algorithm>

#include <iostream>

using namespace deep;

void SDRRL::createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	std::uniform_real_distribution<float> inhibitionDist(initMinInhibition, initMaxInhibition);

	_numStates = numStates;

	_inputs.assign(numStates, 0.0f);
	_reconstructionError.assign(_inputs.size(), 0.0f);

	_cells.resize(numCells);

	_actions.resize(numActions * 2);

	_qConnections.resize(numCells);

	for (int i = 0; i < numCells; i++) {
		_cells[i]._feedForwardConnections.resize(_inputs.size());

		_cells[i]._threshold._weight = initThreshold;

		_cells[i]._actionBias._weight = weightDist(generator);

		for (int j = 0; j < _inputs.size(); j++)
			_cells[i]._feedForwardConnections[j]._weight = weightDist(generator);

		_cells[i]._lateralConnections.resize(_cells.size());

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._lateralConnections[j]._weight = inhibitionDist(generator);

		_cells[i]._actionConnections.resize(_actions.size());

		for (int j = 0; j < _actions.size(); j++)
			_cells[i]._actionConnections[j]._weight = weightDist(generator);

		_qConnections[i]._weight = weightDist(generator);
	}
}

void SDRRL::simStep(float reward, int subIter, float activationLeak, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev); 

	int numHalfActions = _actions.size() / 2;

	for (int i = 0; i < _cells.size(); i++) {
		float excitation = 0.0f;

		for (int j = 0; j < _inputs.size(); j++)
			excitation += _cells[i]._feedForwardConnections[j]._weight * _inputs[j];

		_cells[i]._excitation = excitation;

		_cells[i]._activation = 0.0f;
		_cells[i]._state = 0.0f;
		_cells[i]._spike = 0.0f;
		_cells[i]._spikePrev = 0.0f;
	}

	float subIterInv = 1.0f / subIter;

	for (int iter = 0; iter < subIter; iter++) {
		for (int i = 0; i < _cells.size(); i++) {
			float inhibition = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				if (i != j)
					inhibition += _cells[i]._lateralConnections[j]._weight * _cells[i]._spikePrev;

			float activation = (1.0f - activationLeak) * _cells[i]._activation + _cells[i]._excitation - inhibition;

			if (activation > _cells[i]._threshold._weight) {
				activation = 0.0f;
				_cells[i]._spike = 1.0f;
				_cells[i]._state += subIterInv;
			}
			else
				_cells[i]._spike = 0.0f;

			_cells[i]._activation = activation;
		}

		for (int i = 0; i < _cells.size(); i++)
			_cells[i]._spikePrev = _cells[i]._spike;
	}

	// Use last spike?
	for (int i = 0; i < _cells.size(); i++)
		_cells[i]._state = _cells[i]._spike;

	// Init starting action randomly
	//for (int i = 0; i < _actions.size(); i++) {
	//	_actions[i]._state = dist01(generator);
	//}

	for (int i = 0; i < numHalfActions; i++)
		_actions[i + numHalfActions]._state = 1.0f - _actions[i]._state;

	//std::cout << "Start" << std::endl;

	// Action sampling
	for (int iter = 0; iter < actionDeriveIterations; iter++) {
		float q = 0.0f;

		// Forwards
		for (int k = 0; k < _cells.size(); k++) {
			if (_cells[k]._state > 0.0f) {
				float sum = _cells[k]._actionBias._weight;

				for (int vi = 0; vi < _actions.size(); vi++)
					sum += _cells[k]._actionConnections[vi]._weight * _actions[vi]._state;

				_cells[k]._actionState = sigmoid(sum) * _cells[k]._state;

				q += _qConnections[k]._weight * _cells[k]._actionState;
			}
			else
				_cells[k]._actionState = 0.0f;
		}

		// Action improvement
		for (int k = 0; k < _cells.size(); k++)
			_cells[k]._actionError = _qConnections[k]._weight * _cells[k]._actionState * (1.0f - _cells[k]._actionState);

		for (int i = 0; i < _actions.size(); i++) {
			float sum = 0.0f;

			for (int k = 0; k < _cells.size(); k++)
				sum += _cells[k]._actionConnections[i]._weight * _cells[k]._actionError;

			_actions[i]._error = sum;		
		}

		for (int i = 0; i < numHalfActions; i++)
			// Find action delta
			_actions[i]._state = std::min(1.0f, std::max(0.0f, _actions[i]._state + actionDeriveAlpha * ((_actions[i]._error - _actions[i + numHalfActions]._error) > 0.0f ? 1.0f : -1.0f)));

		for (int i = 0; i < numHalfActions; i++)
			_actions[i + numHalfActions]._state = 1.0f - _actions[i]._state;

		//std::cout << q << std::endl;

	}

	//std::cout << "End" << std::endl;

	// Exploration
	for (int i = 0; i < _actions.size(); i++) {
		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator);
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(0.0f, _actions[i]._state + pertDist(generator)));
	}

	// Forwards
	float q = 0.0f;

	for (int k = 0; k < _cells.size(); k++) {
		if (_cells[k]._state > 0.0f) {
			float sum = _cells[k]._actionBias._weight;

			for (int vi = 0; vi < _actions.size(); vi++)
				sum += _cells[k]._actionConnections[vi]._weight * _actions[vi]._exploratoryState;

			_cells[k]._actionState = sigmoid(sum) * _cells[k]._state;

			q += _qConnections[k]._weight * _cells[k]._actionState;
		}
		else
			_cells[k]._actionState = 0.0f;
	}
	
	//std::cout << q << std::endl;

	float tdError = reward + gamma * q - _prevValue;
	float qAlphaTdError = qAlpha * tdError;
	float actionAlphaTdError = actionAlpha * tdError;
	float surprise = tdError * tdError;

	float learnPattern = sigmoid(surpriseLearnFactor * (surprise - _averageSurprise));
	//std::cout << "LP: " << learnPattern << std::endl;
	_averageSurprise = (1.0f - averageSurpiseDecay) * _averageSurprise + averageSurpiseDecay * surprise;

	// Update weights
	for (int k = 0; k < _cells.size(); k++) {
		float error = _qConnections[k]._weight * _cells[k]._actionState * (1.0f - _cells[k]._actionState);

		_cells[k]._actionBias._weight += actionAlphaTdError * _cells[k]._actionBias._trace;

		_cells[k]._actionBias._trace = _cells[k]._actionBias._trace * gammaLambda + error;

		for (int vi = 0; vi < _actions.size(); vi++) {
			_cells[k]._actionConnections[vi]._weight += actionAlphaTdError * _cells[k]._actionConnections[vi]._trace;

			_cells[k]._actionConnections[vi]._trace = _cells[k]._actionConnections[vi]._trace * gammaLambda + error * _actions[vi]._state;
		}

		_qConnections[k]._weight += qAlphaTdError * _qConnections[k]._trace;

		_qConnections[k]._trace = _qConnections[k]._trace * gammaLambda + _cells[k]._actionState;
	}

	// Reconstruct
	for (int i = 0; i < _reconstructionError.size(); i++) {
		float recon = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			recon += _cells[j]._feedForwardConnections[i]._weight * _cells[j]._state;

		_reconstructionError[i] = (_inputs[i] - recon);
	}

	float sparsitySquared = sparsity * sparsity;

	for (int i = 0; i < _cells.size(); i++) {
		// Learn SDRs
		if (_cells[i]._state > 0.0f) {
			for (int j = 0; j < _inputs.size(); j++)
				_cells[i]._feedForwardConnections[j]._weight += gateFeedForwardAlpha * learnPattern * _cells[i]._state * _reconstructionError[j];// (_inputs[j] - _cells[i]._state * _cells[i]._feedForwardConnections[j]._weight);
		}

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._lateralConnections[j]._weight = std::max(0.0f, _cells[i]._lateralConnections[j]._weight + gateLateralAlpha * learnPattern * (_cells[i]._state * _cells[j]._state - sparsitySquared));

		_cells[i]._threshold._weight += gateBiasAlpha * (_cells[i]._state - sparsity);
	}

	_prevValue = q;
}