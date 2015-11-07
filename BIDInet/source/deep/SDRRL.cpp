#include "SDRRL.h"

#include <algorithm>

#include <iostream>

using namespace deep;

void SDRRL::createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_numStates = numStates;

	_inputs.assign(numStates, 0.0f);
	_reconstructionError.assign(_inputs.size(), 0.0f);

	_cells.resize(numCells);

	_actions.resize(numActions);

	_qConnections.resize(numCells);

	for (int i = 0; i < numCells; i++) {
		_cells[i]._feedForwardConnections.resize(_inputs.size());

		_cells[i]._threshold._weight = initThreshold;

		for (int j = 0; j < _inputs.size(); j++)
			_cells[i]._feedForwardConnections[j]._weight = weightDist(generator);

		_qConnections[i]._weight = weightDist(generator);
	}

	for (int i = 0; i < _actions.size(); i++) {
		_actions[i]._connections.resize(_cells.size());

		for (int k = 0; k < _cells.size(); k++)
			_actions[i]._connections[k]._weight = weightDist(generator);
	}
}

void SDRRL::simStep(float reward, float sparsity, float gamma, int gateSolveIter, float gateFeedForwardAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev); 

	const float numActive = sparsity * _cells.size();

	//for (int iter = 0; iter < gateSolveIter; iter++) {
		for (int i = 0; i < _cells.size(); i++) {
			float excitation = -_cells[i]._threshold._weight;

			for (int j = 0; j < _inputs.size(); j++)
				excitation += _cells[i]._feedForwardConnections[j]._weight * _inputs[j];

			_cells[i]._excitation = excitation;
		}

		for (int i = 0; i < _cells.size(); i++) {
			float numHigher = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				if (i != j)
					if (_cells[j]._excitation >= _cells[i]._excitation)
						numHigher++;

			_cells[i]._state = numHigher < numActive ? 1.0f : 0.0f;
		}

		// Reconstruct
		for (int i = 0; i < _reconstructionError.size(); i++) {
			float recon = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				recon += _cells[j]._feedForwardConnections[i]._weight * _cells[j]._state;

			_reconstructionError[i] = (_inputs[i] - recon);
		}
	//}

	// Forwards
	float q = 0.0f;

	for (int k = 0; k < _cells.size(); k++)
		q += _qConnections[k]._weight * _cells[k]._state;

	for (int a = 0; a < _actions.size(); a++) {
		float sum = 0.0f;

		for (int k = 0; k < _cells.size(); k++)
			sum += _actions[a]._connections[k]._weight * _cells[k]._state;

		_actions[a]._state = sigmoid(sum);
	}

	// Exploration
	for (int a = 0; a < _actions.size(); a++) {
		if (dist01(generator) < explorationBreak)
			_actions[a]._exploratoryState = dist01(generator);
		else
			_actions[a]._exploratoryState = std::min(1.0f, std::max(0.0f, _actions[a]._state + pertDist(generator)));
	}

	float tdError = reward + gamma * q - _prevValue;
	float qAlphaTdError = qAlpha * tdError;
	float actionAlphaTdError = actionAlpha * tdError;
	float surprise = tdError * tdError;

	float learnPattern = sigmoid(surpriseLearnFactor * (surprise - _averageSurprise));
	//std::cout << "LP: " << learnPattern << std::endl;
	_averageSurprise = (1.0f - averageSurpiseDecay) * _averageSurprise + averageSurpiseDecay * surprise;

	// Update weights
	for (int k = 0; k < _cells.size(); k++) {
		_qConnections[k]._weight += qAlphaTdError * _qConnections[k]._trace;

		_qConnections[k]._trace = _qConnections[k]._trace * gammaLambda + _cells[k]._state;
	}

	for (int a = 0; a < _actions.size(); a++) {
		for (int k = 0; k < _cells.size(); k++) {
			_actions[a]._connections[k]._weight += actionAlphaTdError * _actions[a]._connections[k]._trace;

			_actions[a]._connections[k]._trace = _actions[a]._connections[k]._trace * gammaLambda + (_actions[a]._exploratoryState - _actions[a]._state) * _cells[k]._state;
		}
	}

	float sparsitySquared = sparsity * sparsity;

	for (int i = 0; i < _cells.size(); i++) {
		// Learn SDRs
		if (_cells[i]._state > 0.0f) {
			for (int j = 0; j < _inputs.size(); j++)
				_cells[i]._feedForwardConnections[j]._weight += gateFeedForwardAlpha * learnPattern * _cells[i]._state * _reconstructionError[j];
		}

		_cells[i]._threshold._weight += gateBiasAlpha * (_cells[i]._state - sparsity);
	}

	_prevValue = q;
}