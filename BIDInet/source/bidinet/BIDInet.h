#pragma once

#include "../system/ComputeSystem.h"
#include "../system/ComputeProgram.h"

#include <random>
#include <unordered_map>

#include <assert.h>

namespace bidi {
	class BIDInet {
	public:
		enum InputType {
			_state, _action
		};

		struct LayerDesc {
			int _width, _height;

			int _ffRadius, _lRadius, _recRadius, _fbRadius, _predRadius;

			float _ffAlpha, _ffBeta, _ffGamma;
			float _fbPredAlpha, _fbRLAlpha, _fbLambdaGamma;

			float _sparsity;

			LayerDesc()
				: _width(16), _height(16),
				_ffRadius(5), _lRadius(4), _recRadius(3), _fbRadius(5), _predRadius(4),
				_ffAlpha(0.02f), _ffBeta(0.2f), _ffGamma(0.002f),
				_fbPredAlpha(0.2f), _fbRLAlpha(0.05f), _fbLambdaGamma(0.95f),
				_sparsity(0.1f)
			{}
		};

		struct Layer {
			cl::Image2D _ffActivations;

			cl::Image2D _ffStates;
			cl::Image2D _ffStatesPrev;

			cl::Image2D _fbStates;
			cl::Image2D _fbStatesPrev;

			cl::Image2D _ffReconstruction;
			cl::Image2D _recReconstruction;

			cl::Image2D _explorations;
			cl::Image2D _explorationsPrev;

			cl::Image3D _ffConnections;
			cl::Image3D _ffConnectionsPrev;

			cl::Image3D _lConnections;
			cl::Image3D _lConnectionsPrev;

			cl::Image3D _recConnections;
			cl::Image3D _recConnectionsPrev;

			cl::Image3D _fbConnections;
			cl::Image3D _fbConnectionsPrev;

			cl::Image3D _predConnections;
			cl::Image3D _predConnectionsPrev;
		};

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;
		std::vector<InputType> _inputTypes;
		std::vector<float> _inputsTemp;
		std::vector<float> _outputsTemp;

		std::vector<int> _actionIndices;

		int _inputWidth, _inputHeight;

		cl::Image2D _inputs;

		cl::Kernel _ffActivateKernel;
		cl::Kernel _ffInhibitKernel;
		cl::Kernel _fbActivateKernel;
		cl::Kernel _fbActivateFirstKernel;
		cl::Kernel _ffReconstructKernel;
		cl::Kernel _recReconstructKernel;
		cl::Kernel _ffConnectionUpdateKernel;
		cl::Kernel _recConnectionUpdateKernel;
		cl::Kernel _lConnectionUpdateKernel;
		cl::Kernel _fbConnectionUpdateKernel;
		cl::Kernel _predConnectionUpdateKernel;

		float _baseline;

	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		BIDInet()
			: _baseline(0.0f)
		{}

		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, std::mt19937 &generator);

		void simStep(sys::ComputeSystem &cs, float reward, float breakChance, float baselineDecay, std::mt19937 &generator);

		InputType getInputType(int index) const {
			return _inputTypes[index];
		}

		InputType getInputType(int x, int y) const {
			return getInputType(x + y * _inputWidth);
		}

		void setInput(int index, float value) {
			assert(_inputTypes[index] == _state);

			_inputsTemp[index] = value;
		}

		void setInput(int x, int y, float value) {
			_inputsTemp[x + y * _inputWidth] = value;
		}

		float getOutputExploratory(int index) const {
			assert(_inputTypes[index] == _action);

			return _outputsTemp[index];
		}

		float getOutputExploratory(int x, int y) const {
			return getOutputExploratory(x + y * _inputWidth);
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}

		int getInputWidth() const {
			return _inputWidth;
		}

		int getInputHeight() const {
			return _inputHeight;
		}
	};
}