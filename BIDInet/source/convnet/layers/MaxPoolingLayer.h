#pragma once

#include "../Layer.h"

namespace convnet {
	class MaxPoolingLayer : public Layer {
	private:
		int _poolWidth, _poolHeight;

		std::vector<Map> _maxIndices;

	public:
		void create(Layer &input, int width, int height, int numMaps, int poolWidth, int poolHeight);

		virtual void forward(const std::vector<Map> &inputMaps);
		virtual void backward(std::vector<Map> &errorMaps);

		virtual void update(const std::vector<Map> &inputMaps) {}
	};
}