#pragma once

#include <vector>
#include <random>

namespace deep {
	// Unit part of the self-optimizing hierarchy.
	class SelfOptimizingUnit {
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
			std::vector<GateConnection> _gateFeedForwardConnections;
			std::vector<GateConnection> _gateLateralConnections;
			std::vector<StateConnection> _stateConnections;

			GateConnection _gateBias;
			StateConnection _stateBias;

			float _gateActivation;
			float _gate;
			float _state;
			float _statePrev;
			float _error;

			Cell()
				: _statePrev(0.0f)
			{}
		};

		struct Action {
			float _state;
			float _exploratoryState;

			std::vector<StateConnection> _connections;

			StateConnection _bias;

			Action()
				: _state(0.0f), _exploratoryState(0.0f)
			{}
		};

		std::vector<float> _inputs;
		std::vector<float> _reconstruction;
		std::vector<Cell> _cells;
		std::vector<StateConnection> _qConnections;
		std::vector<Action> _actions;

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

		SelfOptimizingUnit()
			: _prevValue(0.0f)
		{}

		void createRandom(int numStates, int numActions, int numCells, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, std::mt19937 &generator);

		void simStep(float reward, float sparsity, float gamma, float gateFeedForwardAlpha, float gateLateralAlpha, float gateBiasAlpha, float stateLinearAlpha, float stateNonlinearAlpha, float actionAlpha, float gammaLambda, float explorationStdDev, float explorationBreak, std::mt19937 &generator);
		
		void setState(int index, float value) {
			_inputs[index] = value;
		}

		float getAction(int index) const {
			return _actions[index]._exploratoryState;
		}

		int getNumStates() const {
			return _inputs.size() - _actions.size();
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


		float getCellGate(int index) const {
			return _cells[index]._gate;
		}
	};
}