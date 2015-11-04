#pragma once

#include "IRSDR.h"
#include "../deep/SDRRL.h"

namespace sdr {
	class IPRSDRRL {
	public:
		enum InputType {
			_state, _action
		};

		struct Connection {
			unsigned short _index;

			float _weightQ;
			float _traceQ;
			float _weightPredictAction;
			float _tracePredictAction;

			Connection()
				: _traceQ(0.0f), _tracePredictAction(0.0f)
			{}
		};

		struct LayerDesc {
			int _width, _height;

			// SDR
			int _receptiveRadius, _recurrentRadius, _lateralRadius, _predictiveRadius, _feedBackRadius;

			float _learnFeedForward, _learnRecurrent, _learnLateral, _learnThreshold;

			int _sdrIter;
			float _sdrStepSize;
			float _sdrLambda;
			float _sdrHiddenDecay;
			float _sdrWeightDecay;
			float _sdrBoostSparsity;
			float _sdrLearnBoost;
			float _sdrNoise;
			float _sdrMaxWeightDelta;

			// SDRRL
			int _cellCount;
			int _rlIter;
			float _rlAlpha;

			float _gateFeedForwardAlpha, _gateThresholdAlpha;
			float _qAlpha, _actionAlpha;

			float _explorationBreakChance;
			float _explorationStdDev;

			float _gamma;
			float _gammaLambda;

			float _averageSurpriseDecay;
			float _surpriseLearnFactor;

			float _driftQ;
			float _driftAction;
			float _cellSparsity;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(6), _recurrentRadius(4), _lateralRadius(4), _predictiveRadius(4), _feedBackRadius(6),
				_learnFeedForward(0.1f), _learnRecurrent(0.1f), _learnLateral(0.2f), _learnThreshold(0.01f),
				_sdrIter(30), _sdrStepSize(0.05f), _sdrLambda(0.3f), _sdrHiddenDecay(0.01f), _sdrWeightDecay(0.001f),
				_sdrBoostSparsity(0.1f), _sdrLearnBoost(0.005f), _sdrNoise(0.01f), _sdrMaxWeightDelta(0.05f),
				_cellCount(8), _rlIter(30), _rlAlpha(0.05f),
				_gateFeedForwardAlpha(0.01f), _gateThresholdAlpha(0.1f),
				_qAlpha(0.007f), _actionAlpha(0.05f),
				_explorationBreakChance(0.01f), _explorationStdDev(0.01f),
				_gamma(0.99f),
				_gammaLambda(0.98f),
				_averageSurpriseDecay(0.01f),
				_surpriseLearnFactor(3.0f),
				_driftQ(0.1f), _driftAction(0.1f),
				_cellSparsity(0.25f)
			{}
		};

		struct PredictionNode {
			deep::SDRRL _sdrrl;

			std::vector<int> _feedBackConnectionIndices;
			std::vector<int> _predictiveConnectionIndices;
		};

		struct Layer {
			IRSDR _sdr;

			std::vector<PredictionNode> _predictionNodes;
		};

		std::vector<PredictionNode> _inputPredictionNodes;

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		std::vector<InputType> _inputTypes;

		std::vector<int> _actionInputIndices;

	public:
		int _cellCount;
		int _rlIter;
		float _rlAlpha;

		float _gateFeedForwardAlpha, _gateThresholdAlpha;
		float _qAlpha, _actionAlpha;

		float _explorationBreakChance;
		float _explorationStdDev;

		float _gamma;
		float _gammaLambda;

		float _averageSurpriseDecay;
		float _surpriseLearnFactor;

		float _driftQ;
		float _driftAction;
		float _cellSparsity;

		float _stateLeak;

		IPRSDRRL()
			: _cellCount(8), _rlIter(30), _rlAlpha(0.05f),
			_gateFeedForwardAlpha(0.01f), _gateThresholdAlpha(0.1f),
			_qAlpha(0.006f), _actionAlpha(0.05f),
			_explorationBreakChance(0.01f), _explorationStdDev(0.01f),
			_gamma(0.99f),
			_gammaLambda(0.98f),
			_averageSurpriseDecay(0.01f),
			_surpriseLearnFactor(3.0f),
			_driftQ(0.1f), _driftAction(0.1f),
			_cellSparsity(0.25f),
			_stateLeak(1.0f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, float initBoost, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator);

		void setState(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value * _stateLeak + (1.0f - _stateLeak) * getAction(index));
		}

		void setState(int x, int y, float value) {
			setState(x + y * _layers.front()._sdr.getVisibleWidth(), value);
		}

		float getActionRel(int index) const {
			return _inputPredictionNodes[_actionInputIndices[index]]._sdrrl.getAction(0);
		}

		float getAction(int index) const {
			return _inputPredictionNodes[index]._sdrrl.getAction(0);
		}

		float getAction(int x, int y) const {
			return getAction(x + y * _layers.front()._sdr.getVisibleWidth());
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}