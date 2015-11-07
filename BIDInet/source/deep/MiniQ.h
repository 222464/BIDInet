#pragma once

#include <random>

#include <vector>

namespace deep {
	class MiniQ {
	public:
		struct Connection {
			float _weight;
			float _trace;

			Connection()
				: _trace(0.0f)
			{}
		};

		struct QNode {
			std::vector<Connection> _connections;

			float _q;
		};

	private:
		std::vector<float> _states;

		std::vector<QNode> _qNodes;

		float _prevValue;

	public:
		MiniQ()
			: _prevValue(0.0f)
		{}

		void createRandom(int numStates, int numQ, float initMinWeight, float initMaxWeight, std::mt19937 &generator);

		// Returns action index
		int simStep(float reward, float alpha, float gamma, float lambdaGamma, float epsilon, std::mt19937 &generator);

		void setState(int index, float value) {
			_states[index] = value;
		}
	};
}