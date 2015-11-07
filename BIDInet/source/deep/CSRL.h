#pragma once

#include "../sdr/IRSDR.h"
#include "MiniQ.h"

namespace deep {
	class CSRL {
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

			int _sdrIterSettle;
			int _sdrIterMeasure;
			float _sdrLeak;
			float _sdrStepSize;
			float _sdrLambda;
			float _sdrHiddenDecay;
			float _sdrWeightDecay;
			float _sparsity;
			float _sdrLearnThreshold;
			float _sdrNoise;
			float _sdrBaselineDecay;
			float _sdrSensitivity;

			float _averageSurpriseDecay;
			float _surpriseLearnFactor;

			int _cellsPerColumn;
			float _cellSparsity;
			float _gamma;
			float _gammaLambda;
			float _gateFeedForwardAlpha;
			float _gateThresholdAlpha;
			int _gateSolveIter;
			float _qAlpha;
			float _actionAlpha, _actionDeriveAlpha;
			int _actionDeriveIterations;
			float _explorationStdDev;
			float _explorationBreak;
					
			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(5), _recurrentRadius(4), _lateralRadius(4), _predictiveRadius(4), _feedBackRadius(5),
				_learnFeedForward(0.005f), _learnRecurrent(0.005f), _learnLateral(0.05f),
				_learnFeedBack(0.1f), _learnPrediction(0.1f),
				_sdrIterSettle(30), _sdrIterMeasure(6), _sdrLeak(0.05f),
				_sdrStepSize(0.1f), _sdrLambda(0.95f), _sdrHiddenDecay(0.01f), _sdrWeightDecay(0.0001f),
				_sparsity(0.1f), _sdrLearnThreshold(0.005f), _sdrNoise(0.05f),
				_sdrBaselineDecay(0.01f), _sdrSensitivity(4.0f),
				_averageSurpriseDecay(0.01f),
				_surpriseLearnFactor(2.0f),
				_cellsPerColumn(16),
				_cellSparsity(0.125f),
				_gamma(0.99f),
				_gammaLambda(0.98f),
				_gateFeedForwardAlpha(0.01f),
				_gateThresholdAlpha(0.005f),
				_gateSolveIter(5),
				_qAlpha(0.01f),
				_actionAlpha(0.05f), _actionDeriveAlpha(0.05f), _actionDeriveIterations(30),
				_explorationStdDev(0.1f), _explorationBreak(0.01f)
			{}
		};

		struct PredictionNode {
			std::vector<Connection> _feedBackConnections;
			std::vector<Connection> _predictiveConnections;

			MiniQ _miniQ;

			unsigned char _action;

			Connection _bias;

			float _state;
			float _statePrev;

			float _stateOutput;
			float _stateOutputPrev;

			PredictionNode()
				: _action(0), _state(0.0f), _statePrev(0.0f), _stateOutput(0.0f), _stateOutputPrev(0.0f)
			{}
		};

		struct InputPredictionNode {
			std::vector<Connection> _feedBackConnections;

			MiniQ _miniQ;

			unsigned char _action;

			Connection _bias;

			float _state;
			float _statePrev;

			float _stateOutput;
			float _stateOutputPrev;
			InputPredictionNode()
				: _action(0), _state(0.0f), _statePrev(0.0f), _stateOutput(0.0f), _stateOutputPrev(0.0f)
			{}
		};

		struct Layer {
			sdr::IRSDR _sdr;

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
		float _learnFeedBack;

		float _averageSurpriseDecay;
		float _surpriseLearnFactor;

		int _cellsPerColumn;
		float _cellSparsity;
		float _gamma;
		float _gammaLambda;
		float _gateFeedForwardAlpha;
		float _gateThresholdAlpha;
		int _gateSolveIter;
		float _qAlpha;
		float _actionAlpha, _actionDeriveAlpha;
		int _actionDeriveIterations;
		float _explorationStdDev;
		float _explorationBreak;

		CSRL()
			: _learnFeedBack(0.1f),
			_averageSurpriseDecay(0.01f),
			_surpriseLearnFactor(2.0f),
			_cellsPerColumn(16),
			_cellSparsity(0.125f),
			_gamma(0.99f),
			_gammaLambda(0.98f),
			_gateFeedForwardAlpha(0.05f),
			_gateThresholdAlpha(0.005f),
			_gateSolveIter(5),
			_qAlpha(0.01f),
			_actionAlpha(0.05f), _actionDeriveAlpha(0.05f), _actionDeriveIterations(30),
			_explorationStdDev(0.1f), _explorationBreak(0.01f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator, bool learn = true);

		void setInput(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value);
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _layers.front()._sdr.getVisibleWidth(), value);
		}

		float getPrediction(int index) const {
			return _inputPredictionNodes[index]._stateOutput;
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