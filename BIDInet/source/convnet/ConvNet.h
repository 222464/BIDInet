#pragma once

#include "Layer.h"

#include <memory>

namespace convnet {
	class ConvNet {
	public:
		struct Connection {
			float _weight;
			float _prevDWeight;

			Connection()
				: _prevDWeight(0.0f)
			{}
		};

		struct Node {
			float _output;
			float _error;

			std::vector<Connection> _connections;

			Connection _bias;

			Node()
				: _output(0.0f)
			{}
		};

	private:
		std::vector<std::shared_ptr<Layer>> _layers;

		std::vector<Node> _hiddenNodes;
		std::vector<Node> _outputNodes;

	public:
		float _reluLeak;
		float _hiddenAlpha;
		float _outputAlpha;
		float _hiddenMomentum;
		float _outputMomentum;

		ConvNet()
			: _reluLeak(0.01f),
			_hiddenAlpha(0.1f),
			_outputAlpha(0.02f),
			_hiddenMomentum(0.0f),
			_outputMomentum(0.0f)
		{}

		// Call after adding layers
		void create(int numHidden, int numOutputs, float initMinWeight, float initMaxWeight, std::mt19937 &generator);

		float getOutput(int index) const {
			return _outputNodes[index]._output;
		}

		void setError(int index, float error) {
			_outputNodes[index]._error = error;
		}

		void addLayer(const std::shared_ptr<Layer> &layer) {
			_layers.push_back(layer);
		}

		void forward();
		void backward();
		void update();

		int getNumOutputs() const {
			return _outputNodes.size();
		}

		int getNumLayers() const {
			return _layers.size();
		}

		const std::shared_ptr<Layer> &getLayer(int index) const {
			return _layers[index];
		}
	};
}