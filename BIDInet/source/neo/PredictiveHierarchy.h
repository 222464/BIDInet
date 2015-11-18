#pragma once

#include "SparseCoder.h"

namespace neo {
	class PredictiveHierarchy {
	public:
		struct Connection {
			unsigned short _index;

			float _weight;
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveRadius, _recurrentRadius, _lateralRadius, _predictiveRadius, _feedBackRadius;

			float _learnFeedForward, _learnRecurrent, _learnLateral;

			float _learnFeedBack, _learnPrediction;

			int _sdrIter;
			float _sdrLeak;
			float _sdrLambda;
			float _sdrHiddenDecay;
			float _sdrWeightDecay;
			float _sdrMaxWeightDelta;
			float _sdrSparsity;
			float _sdrLearnThreshold;
			float _sdrBaselineDecay;
			float _sdrSensitivity;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(4), _recurrentRadius(4), _lateralRadius(4), _predictiveRadius(4), _feedBackRadius(4),
				_learnFeedForward(0.01f), _learnRecurrent(0.01f), _learnLateral(0.05f),
				_learnFeedBack(0.1f), _learnPrediction(0.03f),
				_sdrIter(30),
				_sdrLeak(0.1f), _sdrLambda(0.95f), _sdrHiddenDecay(0.01f), _sdrWeightDecay(0.0f), _sdrMaxWeightDelta(0.5f),
				_sdrSparsity(0.02f), _sdrLearnThreshold(0.01f),
				_sdrBaselineDecay(0.01f),
				_sdrSensitivity(6.0f)
			{}
		};

		struct PredictionNode {
			std::vector<Connection> _feedBackConnections;
			std::vector<Connection> _predictiveConnections;

			Connection _bias;

			float _state;
			float _statePrev;

			float _activation;
			float _activationPrev;

			float _baseline;

			PredictionNode()
				: _state(0.0f), _statePrev(0.0f), _activation(0.0f), _activationPrev(0.0f), _baseline(0.0f)
			{}
		};

		struct InputPredictionNode {
			std::vector<Connection> _feedBackConnections;

			Connection _bias;

			float _state;
			float _statePrev;

			float _activation;
			float _activationPrev;

			InputPredictionNode()
				: _state(0.0f), _statePrev(0.0f), _activation(0.0f), _activationPrev(0.0f)
			{}
		};

		struct Layer {
			SparseCoder _sdr;

			std::vector<PredictionNode> _predictionNodes;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		std::vector<InputPredictionNode> _inputPredictionNodes;


	public:
		float _learnInputFeedBack;

		PredictiveHierarchy()
			: _learnInputFeedBack(0.1f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator);

		void simStep(std::mt19937 &generator, bool learn = true);

		void simStepGenerate(std::mt19937 &generator, float noise);

		void setInput(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value);
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _layerDescs.front()._width, value);
		}

		float getPrediction(int index) const {
			return _inputPredictionNodes[index]._state;
		}

		float getPrediction(int x, int y) const {
			return getPrediction(x + y * _layers.front()._sdr.getVisibleWidth());
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}