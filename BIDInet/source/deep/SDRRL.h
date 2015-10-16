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
		};

		struct Cell {
			std::vector<GateConnection> _feedForwardConnections;
			std::vector<GateConnection> _lateralConnections;
	
			GateConnection _bias;

			float _activation;
			float _state;
			float _statePrev;

			float _actionState;
			float _actionError;
			
			float _trace;

			StateConnection _tdConnection;

			Cell()
				: _statePrev(0.0f), _trace(0.0f)
			{}
		};

		struct Action {
			float _state;
			float _exploratoryState;
			float _error;

			std::vector<StateConnection> _connections;

			Action()
				: _state(0.0f), _exploratoryState(0.0f), _error(0.0f)
			{}
		};

		std::vector<float> _inputs;
		std::vector<float> _reconstructionError;
		std::vector<Cell> _cells;
		std::vector<StateConnection> _qConnections;
		std::vector<Action> _actions;

		int _numStates;

		float _prevValue;

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
			: _prevValue(0.0f)
		{}

		void createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, std::mt19937 &generator);

		void simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float qAlpha, float actionAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, std::mt19937 &generator);
		
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