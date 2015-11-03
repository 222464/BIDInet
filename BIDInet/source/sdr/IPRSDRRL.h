#pragma once

#include "IRSDR.h"

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

			int _receptiveRadius, _recurrentRadius, _lateralRadius, _predictiveRadius, _feedBackRadius;

			float _learnFeedForward, _learnRecurrent, _learnLateral, _learnThreshold;

			float _learnFeedBackPred, _learnPredictionPred;
			float _learnFeedBackAction, _learnPredictionAction;
			float _learnFeedBackQ, _learnPredictionQ;

			float _exploratoryNoiseChance;
			float _exploratoryNoise;

			int _sdrIter;
			float _sdrStepSize;
			float _sdrLambda;
			float _sdrHiddenDecay;
			float _sdrWeightDecay;
			float _sdrBoostSparsity;
			float _sdrLearnBoost;
			float _sdrNoise;
			float _sdrMaxWeightDelta;

			float _gamma;
			float _gammaLambda;

			float _averageSurpriseDecay;
			float _attentionFactor;

			float _sparsity;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(8), _recurrentRadius(6), _lateralRadius(5), _predictiveRadius(6), _feedBackRadius(8),
				_learnFeedForward(0.01f), _learnRecurrent(0.01f), _learnLateral(0.2f), _learnThreshold(0.01f),
				_learnFeedBackPred(0.05f), _learnPredictionPred(0.05f),
				_learnFeedBackAction(0.1f), _learnPredictionAction(0.1f),
				_learnFeedBackQ(0.1f), _learnPredictionQ(0.1f),
				_exploratoryNoiseChance(0.01f), _exploratoryNoise(0.05f),
				_sdrIter(30), _sdrStepSize(0.05f), _sdrLambda(0.3f), _sdrHiddenDecay(0.01f), _sdrWeightDecay(0.001f),
				_sdrBoostSparsity(0.1f), _sdrLearnBoost(0.02f), _sdrNoise(0.01f), _sdrMaxWeightDelta(0.05f),
				_gamma(0.99f),
				_gammaLambda(0.98f),
				_averageSurpriseDecay(0.01f),
				_attentionFactor(4.0f),
				_sparsity(0.01f)
			{}
		};

		struct PredictionNode {
			std::vector<Connection> _feedBackConnections;
			std::vector<Connection> _predictiveConnections;

			Connection _bias;

			float _action;
			float _actionPrev;
			
			float _actionExploratory;
			float _actionExploratoryPrev;

			float _q;
			float _qPrev;

			float _averageSurprise; // Use to keep track of importance for prediction. If current error is greater than average, then attention is > 0.5 else < 0.5 (sigmoid)

			PredictionNode()
				: _action(0.0f), _actionPrev(0.0f), _actionExploratory(0.0f), _actionExploratoryPrev(0.0f),
				_q(0.0f), _qPrev(0.0f), _averageSurprise(0.0f)
			{}
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

		float _prevValue;

	public:
		float _stateLeak;
		float _exploratoryNoiseChance;
		float _exploratoryNoise;
		float _gamma;
		float _gammaLambda;
		float _actionRandomizeChance;
		float _qAlpha;
		float _learnFeedBackPred;
		float _learnFeedBackAction;
		float _learnFeedBackQ;
		float _driftLearn;

		IPRSDRRL()
			: _prevValue(0.0f),
			_stateLeak(1.0f),
			_exploratoryNoiseChance(0.05f),
			_exploratoryNoise(0.05f),
			_gamma(0.99f),
			_gammaLambda(0.95f),
			_actionRandomizeChance(0.01f),
			_qAlpha(0.6f),
			_learnFeedBackPred(0.05f),
			_learnFeedBackAction(0.1f),
			_learnFeedBackQ(0.1f),
			_driftLearn(0.2f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator);

		void setState(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value * _stateLeak + (1.0f - _stateLeak) * getAction(index));
		}

		void setState(int x, int y, float value) {
			setState(x + y * _layers.front()._sdr.getVisibleWidth(), value);
		}

		float getActionRel(int index) const {
			return _inputPredictionNodes[_actionInputIndices[index]]._actionExploratory;
		}

		float getAction(int index) const {
			return _inputPredictionNodes[index]._actionExploratory;
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