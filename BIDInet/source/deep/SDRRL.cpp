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

	_actions.resize(numActions);

	_qConnections.resize(numCells);

	for (int i = 0; i < numCells; i++) {
		_cells[i]._feedForwardConnections.resize(_inputs.size());

		_cells[i]._lateralConnections.resize(numCells);

		_cells[i]._threshold = initThreshold;

		for (int j = 0; j < _inputs.size(); j++)
			_cells[i]._feedForwardConnections[j]._weight = weightDist(generator);

		for (int j = 0; j < numCells; j++)
			_cells[i]._lateralConnections[j]._weight = inhibitionDist(generator);

		_qConnections[i]._weight = weightDist(generator);
	}

	for (int i = 0; i < _actions.size(); i++) {
		_actions[i]._connections.resize(_cells.size());

		for (int k = 0; k < _cells.size(); k++)
			_actions[i]._connections[k]._weight = weightDist(generator);
	}
}

void SDRRL::simStep(float reward, float sparsity, float gamma, int subIterSettle, int subIterMeasure, float leak, float gateFeedForwardAlpha, float gateLateralAlpha, float gateThresholdAlpha, float qAlpha, float actionAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev); 

	// Clear activations and states
	for (int i = 0; i < _cells.size(); i++) {
		_cells[i]._activation = 0.0f;
		_cells[i]._state = 0.0f;
	}

	for (int iter = 0; iter < subIterSettle; iter++) {
		// Activate
		for (int i = 0; i < _cells.size(); i++) {
			float excitation = 0.0f;

			for (int j = 0; j < _inputs.size(); j++)
				excitation += _cells[i]._feedForwardConnections[j]._weight * _reconstructionError[j];

			float inhibition = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				inhibition += _cells[i]._lateralConnections[j]._weight * _cells[j]._spikePrev;

			float activation = (1.0f - leak) * _cells[i]._activation + excitation - inhibition;

			if (activation > _cells[i]._threshold) {
				activation = 0.0f;

				_cells[i]._spike = 1.0f;
			}
			else
				_cells[i]._spike = 0.0f;

			_cells[i]._activation = activation;
		}

		// Double buffer update
		for (int i = 0; i < _cells.size(); i++)
			_cells[i]._spikePrev = _cells[i]._spike;

		// Reconstruct
		for (int i = 0; i < _inputs.size(); i++) {
			float recon = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				recon += _cells[j]._spike * _cells[j]._feedForwardConnections[i]._weight;

			_reconstructionError[i] = _inputs[i] - recon;
		}
	}

	const float subIterMeasureInv = 1.0f / subIterMeasure;

	for (int iter = 0; iter < subIterMeasure; iter++) {
		// Activate
		for (int i = 0; i < _cells.size(); i++) {
			float excitation = 0.0f;

			for (int j = 0; j < _inputs.size(); j++)
				excitation += _cells[i]._feedForwardConnections[j]._weight * _reconstructionError[j];

			float inhibition = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				inhibition += _cells[i]._lateralConnections[j]._weight * _cells[j]._spikePrev;

			float activation = (1.0f - leak) * _cells[i]._activation + excitation - inhibition;

			if (activation > _cells[i]._threshold) {
				activation = 0.0f;

				_cells[i]._spike = 1.0f;
			}
			else
				_cells[i]._spike = 0.0f;

			_cells[i]._state += _cells[i]._spike * subIterMeasureInv;

			_cells[i]._activation = activation;
		}

		// Double buffer update
		for (int i = 0; i < _cells.size(); i++)
			_cells[i]._spikePrev = _cells[i]._spike;

		// Reconstruct
		for (int i = 0; i < _inputs.size(); i++) {
			float recon = 0.0f;

			for (int j = 0; j < _cells.size(); j++)
				recon += _cells[j]._spike * _cells[j]._feedForwardConnections[i]._weight;

			_reconstructionError[i] = _inputs[i] - recon;
		}
	}

	// Final state reconstruction
	for (int i = 0; i < _inputs.size(); i++) {
		float recon = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			recon += _cells[j]._state * _cells[j]._feedForwardConnections[i]._weight;

		_reconstructionError[i] = _inputs[i] - recon;
	}

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
	
	_averageSurprise = (1.0f - averageSurpiseDecay) * _averageSurprise + averageSurpiseDecay * surprise;

	// Update weights
	for (int k = 0; k < _cells.size(); k++) {
		_qConnections[k]._weight += qAlphaTdError * _qConnections[k]._trace;

		_qConnections[k]._trace = _qConnections[k]._trace * gammaLambda + _cells[k]._state;
	}

	for (int a = 0; a < _actions.size(); a++) {
		for (int i = 0; i < _cells.size(); i++) {
			_actions[a]._connections[i]._weight += actionAlphaTdError * _actions[a]._connections[i]._trace;

			_actions[a]._connections[i]._trace = _actions[a]._connections[i]._trace * gammaLambda + (_actions[a]._exploratoryState - _actions[a]._state) * _cells[i]._state;
		}
	}

	float sparsitySquared = sparsity * sparsity;

	for (int i = 0; i < _cells.size(); i++) {
		// Learn SDRs
		if (_cells[i]._state > 0.0f) {
			for (int j = 0; j < _inputs.size(); j++)
				_cells[i]._feedForwardConnections[j]._weight += gateFeedForwardAlpha * _cells[i]._state * _reconstructionError[j];
		}

		for (int j = 0; j < _cells.size(); j++)
			_cells[i]._lateralConnections[j]._weight = std::max(0.0f, _cells[i]._lateralConnections[j]._weight + gateLateralAlpha * (_cells[i]._state * _cells[j]._state - sparsitySquared));

		_cells[i]._threshold += gateThresholdAlpha * (_cells[i]._state - sparsity);
	}

	_prevValue = q;
}