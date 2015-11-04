#pragma once

#include <vector>
#include <random>

namespace deep {
	// Unit part of the self-optimizing hierarchy.
	class SDRRL {
	private:
		struct GateConnection {
			float _weight;
		};

		struct StateConnection {
			float _weight;
			float _trace;

			StateConnection()
				: _trace(0.0f)
			{}
		};

		struct Cell {
			std::vector<GateConnection> _feedForwardConnections;
			std::vector<StateConnection> _actionConnections;

			GateConnection _threshold;

			StateConnection _actionBias;

			float _excitation;
			float _state;
			float _statePrev;

			float _actionState;
			float _actionError;

			Cell()
				: _statePrev(0.0f)
			{}
		};

		struct Action {
			float _state;
			float _statePrev;
			float _exploratoryState;
			float _error;
	
			Action()
				: _state(0.0f), _statePrev(0.0f), _exploratoryState(0.0f)
			{}
		};

		std::vector<float> _inputs;
		std::vector<float> _reconstructionError;
		std::vector<Cell> _cells;
		std::vector<StateConnection> _qConnections;
		std::vector<Action> _actions;

		int _numStates;

		float _prevValue;
		float _averageSurprise;

	public:
		static float relu(float x, float leak) {
			return x > 0.0f ? x : x * leak;
		}

		static float relud(float x, float leak) {
			return x > 0.0f ? 1.0f : leak;
		}

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		SDRRL()
			: _prevValue(0.0f), _averageSurprise(0.0f)
		{}

		void createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator);

		void simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateThresholdAlpha, float qAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator);
		void simStepDrift(const std::vector<float> &targets, float driftQ, float driftAction, float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateThresholdAlpha, float qAlpha, float actionAlpha, int actionDeriveIterations, float actionDeriveAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, float averageSurpiseDecay, float surpriseLearnFactor, std::mt19937 &generator);

		void setState(int index, float value) {
			_inputs[index] = value;
		}

		float getAction(int index) const {
			return _actions[index]._exploratoryState;
		}

		int getNumStates() const {
			return _inputs.size();
		}

		int getNumActions() const {
			return _actions.size();
		}

		int getNumCells() const {
			return _cells.size();
		}

		float getCellState(int index) const {
			return _cells[index]._state;
		}
	};
}