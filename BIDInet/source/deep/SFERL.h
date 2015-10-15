#pragma once

#include <vector>
#include <list>
#include <random>
#include <string>

namespace deep {
	class SFERL {
	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		struct Connection {
			float _weight;
			float _trace;

			Connection()
				: _trace(0.0f)
			{}
		};

		struct Hidden {
			Connection _bias;

			std::vector<Connection> _feedForwardConnections;
			std::vector<float> _lateralConnections; 

			float _activation;
			float _state;

			Hidden()
				: _activation(0.0f), _state(0.0f)
			{}
		};

		struct Visible {
			Connection _bias;

			float _state;

			Visible()
				: _state(0.0f)
			{}
		};

		struct Action {
			Connection _bias;

			std::vector<Connection> _feedForwardConnections;

			float _state;

			Action()
				: _state(0.0f)
			{}
		};

		std::vector<Hidden> _hidden;
		std::vector<Visible> _visible;
		std::vector<Action> _actions;

		int _numState;
		int _numAction;

		float _zInv;

		float _prevValue;

		std::vector<float> _prevVisible;
		std::vector<float> _prevHidden;

	public:
		SFERL();

		void createRandom(int numState, int numAction, int numHidden, float weightStdDev, float maxInhibition, std::mt19937 &generator);

		// Returns action index
		void step(const std::vector<float> &state, std::vector<float> &action,
			float reward, float gamma, float lambdaGamma, float actionSearchAlpha,
			float lateralAlpha, float actionAlpha, float gradientAlpha, float sparsitySquared,
			float breakChance, float perturbationStdDev,
			std::mt19937 &generator);

		void activate();
		void updateOnError(float error, float lateralAlpha, float sparsitySquared, float lambdaGamma);

		float freeEnergy() const;

		float value() const {
			return -freeEnergy() * _zInv;
		}

		int getNumState() const {
			return _numState;
		}

		int getNumAction() const {
			return _numAction;
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		float getHiddenState(int index) const {
			return _hidden[index]._state;
		}

		float getZInv() const {
			return _zInv;
		}
	};
}