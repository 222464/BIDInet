#pragma once

#include "../Layer.h"

namespace convnet {
	class ConvLayer : public Layer {
	private:
		struct Connection {
			float _weight;
		};

		struct Kernel {
			std::vector<Connection> _connections;
		};

		std::vector<Kernel> _mapKernels;

		int _convWidth, _convHeight, _convNumMaps;

	public:
		float _reluLeak;

		float _alpha;

		ConvLayer()
			: _reluLeak(0.01f), _alpha(0.001f)
		{}

		void create(Layer &input, int width, int height, int numMaps, int convWidth, int convHeight,
			float initMinWeight, float initMaxWeight, std::mt19937 &generator);

		virtual void forward(const std::vector<Map> &inputMaps);
		virtual void backward(std::vector<Map> &errorMaps);

		virtual void update(const std::vector<Map> &inputMaps);
	};
}