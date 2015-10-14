#pragma once

#include "../Layer.h"

namespace convnet {
	class InputLayer : public Layer {
	public:
		void create(int width, int height, int numMaps) {
			_outputMaps.resize(numMaps);
			_errorMaps.resize(numMaps);

			for (int m = 0; m < numMaps; m++) {
				_outputMaps[m].create(width, height);
				_errorMaps[m].create(width, height);
			}
		}

		virtual void forward(const std::vector<Map> &inputMaps) {}
		virtual void backward(std::vector<Map> &errorMaps) {}

		virtual void update(const std::vector<Map> &inputMaps) {}
	};
}