#pragma once

#include "Map.h"

namespace convnet {
	class Layer {
	protected:
		std::vector<Map> _outputMaps;
		std::vector<Map> _errorMaps;

	public:
		virtual ~Layer() {}

		virtual void forward(const std::vector<Map> &inputMaps) = 0;
		virtual void backward(std::vector<Map> &errorMaps) = 0;

		virtual void update(const std::vector<Map> &inputMaps) = 0;

		int getNumMaps() const {
			return _outputMaps.size();
		}

		int getOutputWidth() const {
			return _outputMaps.front().getWidth();
		}

		int getOutputHeight() const {
			return _outputMaps.front().getHeight();
		}

		int getErrorWidth() const {
			return _errorMaps.front().getWidth();
		}

		int getErrorHeight() const {
			return _errorMaps.front().getHeight();
		}

		std::vector<Map> &getOutputMaps() {
			return _outputMaps;
		}

		std::vector<Map> &getErrorMaps() {
			return _errorMaps;
		}

		friend class ConvNet;
	};

	float relu(float x, float leak);

	float relud(float x, float leak);
}