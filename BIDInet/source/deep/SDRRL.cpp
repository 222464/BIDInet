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

	_actions.resize(numActions * 2);

	_qConnections.resize(numCells);

	for (int i = 0; i < numCells; i++) {
		_cells[i]._feedForwardConnections.resize(_inputs.size());

		_cells[i]._threshold._weight = initThreshold;

		_cells[i]._actionBias._weight = weightDist(generator);

		for (int j = 0; j < _inputs.size(); j++)
			_cells[i]._feedForwardConnections[j]._weight = weightDist(generator);

		_cells[i]._actionConnections.resize(_actions.size());

		for (int j = 0; j < _actions.size(); j++)
			_cells[i]._actionConnections[j]._weight = weightDist(generator);

		_qConnections[i]._weight = weightDist(generator);
	}
}

void SDRRL::simStep(float reward, float sparsity, float gamma, int gateSolveIter, float gateFeedForwardAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev); 

	int numHalfActions = _actions.size() / 2;

	const float numActive = sparsity * _cells.size();

	for (int iter = 0; iter < gateSolveIter; iter++) {
		for (int i = 0; i < _cells.size(); i++) {
			float excitation = -_cells[i]._threshold._weight;

			for (int j = 0; j < _inputs.size(); j++)
				excitation += _cells[i]._feedForwardConnections[j]._weight * _reconstructionError[j];

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
	}

	// Init starting action randomly
	//for (int i = 0; i < _actions.size(); i++) {
	//	_actions[i]._state = dist01(generator);
	//}

	for (int i = 0; i < numHalfActions; i++)
		_actions[i + numHalfActions]._state = 1.0f - _actions[i]._state;

	// Action sampling
	for (int iter = 0; iter < actionDeriveIterations; iter++) {
		float q = 0.0f;

		// Forwards
		for (int k = 0; k < _cells.size(); k++) {
			if (_cells[k]._state > 0.0f) {
				float sum = 0.0f;// _cells[k]._actionBias._weight;

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
	}

	// Exploration
	for (int i = 0; i < _actions.size(); i++) {
		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator);
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(0.0f, _actions[i]._state + pertDist(generator)));
	}

	for (int i = 0; i < numHalfActions; i++)
		_actions[i + numHalfActions]._exploratoryState = 1.0f - _actions[i]._exploratoryState;

	// Forwards
	float q = 0.0f;

	for (int k = 0; k < _cells.size(); k++) {
		if (_cells[k]._state > 0.0f) {
			float sum = 0.0f;// _cells[k]._actionBias._weight;

			for (int vi = 0; vi < _actions.size(); vi++)
				sum += _cells[k]._actionConnections[vi]._weight * _actions[vi]._exploratoryState;

			_cells[k]._actionState = sigmoid(sum) * _cells[k]._state;

			q += _qConnections[k]._weight * _cells[k]._actionState;
		}
		else
			_cells[k]._actionState = 0.0f;
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
		float error = _qConnections[k]._weight * _cells[k]._actionState * (1.0f - _cells[k]._actionState);

		_cells[k]._actionBias._weight += actionAlphaTdError * _cells[k]._actionBias._trace;

		_cells[k]._actionBias._trace = _cells[k]._actionBias._trace * gammaLambda + error;

		for (int vi = 0; vi < _actions.size(); vi++) {
			_cells[k]._actionConnections[vi]._weight += actionAlphaTdError * _cells[k]._actionConnections[vi]._trace;

			_cells[k]._actionConnections[vi]._trace = _cells[k]._actionConnections[vi]._trace * gammaLambda + error * _actions[vi]._exploratoryState;
		}

		_qConnections[k]._weight += qAlphaTdError * _qConnections[k]._trace;

		_qConnections[k]._trace = _qConnections[k]._trace * gammaLambda + _cells[k]._actionState;
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

void SDRRL::simStepDrift(const std::vector<float> &targets, float driftQ, float driftAction, float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, explorationStdDev);

	int numHalfActions = _actions.size() / 2;

	std::vector<float> fullTargets(_actions.size());

	for (int i = 0; i < numHalfActions; i++)
		fullTargets[i] = targets[i];

	for (int i = 0; i < numHalfActions; i++)
		fullTargets[i + numHalfActions] = 1.0f - targets[i];

	for (int i = 0; i < _cells.size(); i++) {
		float excitation = -_cells[i]._threshold._weight;

		for (int j = 0; j < _inputs.size(); j++)
			excitation += _cells[i]._feedForwardConnections[j]._weight * _inputs[j];

		_cells[i]._excitation = excitation;
	}

	const float numActive = sparsity * _cells.size();

	for (int i = 0; i < _cells.size(); i++) {
		float numHigher = 0.0f;

		for (int j = 0; j < _cells.size(); j++)
			if (i != j)
				if (_cells[j]._excitation >= _cells[i]._excitation)
					numHigher++;

		_cells[i]._state = numHigher < numActive ? 1.0f : 0.0f;
	}

	// Init starting action randomly
	//for (int i = 0; i < _actions.size(); i++) {
	//	_actions[i]._state = dist01(generator);
	//}

	for (int i = 0; i < numHalfActions; i++)
		_actions[i + numHalfActions]._state = 1.0f - _actions[i]._state;

	// Action sampling
	for (int iter = 0; iter < actionDeriveIterations; iter++) {
		float q = 0.0f;

		// Forwards
		for (int k = 0; k < _cells.size(); k++) {
			if (_cells[k]._state > 0.0f) {
				float sum = 0.0f;// _cells[k]._actionBias._weight;

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
	}

	// Exploration
	for (int i = 0; i < numHalfActions; i++) {
		if (dist01(generator) < explorationBreak)
			_actions[i]._exploratoryState = dist01(generator);
		else
			_actions[i]._exploratoryState = std::min(1.0f, std::max(-1.0f, _actions[i]._state + pertDist(generator)));
	}

	for (int i = 0; i < numHalfActions; i++)
		_actions[i + numHalfActions]._exploratoryState = 1.0f - _actions[i]._exploratoryState;

	// Forwards
	float q = 0.0f;

	for (int k = 0; k < _cells.size(); k++) {
		if (_cells[k]._state > 0.0f) {
			float sum = 0.0f;// _cells[k]._actionBias._weight;

			for (int vi = 0; vi < _actions.size(); vi++)
				sum += _cells[k]._actionConnections[vi]._weight * _actions[vi]._exploratoryState;

			_cells[k]._actionState = sigmoid(sum) * _cells[k]._state;

			q += _qConnections[k]._weight * _cells[k]._actionState;
		}
		else
			_cells[k]._actionState = 0.0f;
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
		float error = _qConnections[k]._weight * _cells[k]._actionState * (1.0f - _cells[k]._actionState);

		_cells[k]._actionBias._weight += actionAlphaTdError * _cells[k]._actionBias._trace;

		_cells[k]._actionBias._trace = _cells[k]._actionBias._trace * gammaLambda + error;

		for (int vi = 0; vi < _actions.size(); vi++) {
			_cells[k]._actionConnections[vi]._weight += actionAlphaTdError * _cells[k]._actionConnections[vi]._trace;

			_cells[k]._actionConnections[vi]._trace = _cells[k]._actionConnections[vi]._trace * gammaLambda + error * _actions[vi]._exploratoryState;
		}

		_qConnections[k]._weight += qAlphaTdError * _qConnections[k]._trace;

		_qConnections[k]._trace = _qConnections[k]._trace * gammaLambda + _cells[k]._actionState;
	}

	// Drag previous actions to prediction targets while keeping Q constant
	float prevPredictionQ = 0.0f;

	for (int k = 0; k < _cells.size(); k++) {
		if (_cells[k]._statePrev > 0.0f) {
			float sum = 0.0f;// _cells[k]._actionBias._weight;

			for (int vi = 0; vi < _actions.size(); vi++)
				sum += _cells[k]._actionConnections[vi]._weight * fullTargets[vi];

			_cells[k]._actionState = sigmoid(sum) * _cells[k]._statePrev;

			prevPredictionQ += _qConnections[k]._weight * _cells[k]._actionState;
		}
		else
			_cells[k]._actionState = 0.0f;
	}

	for (int k = 0; k < _cells.size(); k++) {
		float error = (_prevValue - prevPredictionQ) * _qConnections[k]._weight * _cells[k]._actionState * (1.0f - _cells[k]._actionState);

		_cells[k]._actionBias._weight += driftAction * error;

		for (int vi = 0; vi < _actions.size(); vi++)
			_cells[k]._actionConnections[vi]._weight += driftAction * error * fullTargets[vi];

		_qConnections[k]._weight += driftQ * (_prevValue - prevPredictionQ) * _cells[k]._actionState;
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
				_cells[i]._feedForwardConnections[j]._weight += gateFeedForwardAlpha * learnPattern * _cells[i]._state * _reconstructionError[j];
		}

		_cells[i]._threshold._weight += gateBiasAlpha * (_cells[i]._state - sparsity);
	}

	for (int i = 0; i < _cells.size(); i++)
		_cells[i]._statePrev = _cells[i]._state;

	for (int i = 0; i < _actions.size(); i++)
		_actions[i]._statePrev = _actions[i]._state;

	_prevValue = q;
}