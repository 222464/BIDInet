#pragma once

#include "IRSDR.h"

namespace sdr {
	class IPRSDRRL {
	public:
		enum InputType {
			_state, _q, _action
		};

		struct Connection {
			unsigned short _index;

			float _weight;
			float _trace;
			float _tracePrev;

			Connection()
				: _trace(0.0f), _tracePrev(0.0f)
			{}
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveRadius, _recurrentRadius, _lateralRadius, _predictiveRadius, _feedBackRadius;

			float _learnFeedForward, _learnRecurrent, _learnLateral, _learnThreshold;

			float _learnFeedBackPred, _learnPredictionPred;
			float _learnFeedBackRL, _learnPredictionRL;

			int _sdrIter;
			float _sdrStepSize;
			float _sdrLambda;
			float _sdrHiddenDecay;
			float _sdrWeightDecay;
			float _sdrBoostSparsity;
			float _sdrLearnBoost;
			float _sdrNoise;

			float _averageSurpriseDecay;
			float _attentionFactor;

			float _sparsity;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(12), _recurrentRadius(6), _lateralRadius(5), _predictiveRadius(6), _feedBackRadius(12),
				_learnFeedForward(0.05f), _learnRecurrent(0.05f), _learnLateral(0.2f), _learnThreshold(0.12f),
				_learnFeedBackPred(0.5f), _learnPredictionPred(0.5f),
				_learnFeedBackRL(0.1f), _learnPredictionRL(0.1f),
				_sdrIter(30), _sdrStepSize(0.05f), _sdrLambda(0.4f), _sdrHiddenDecay(0.01f), _sdrWeightDecay(0.0001f),
				_sdrBoostSparsity(0.02f), _sdrLearnBoost(0.05f), _sdrNoise(0.01f),
				_averageSurpriseDecay(0.01f),
				_attentionFactor(4.0f),
				_sparsity(0.01f)
			{}
		};

		struct PredictionNode {
			std::vector<Connection> _feedBackConnections;
			std::vector<Connection> _predictiveConnections;

			Connection _bias;

			float _state;
			float _statePrev;
			
			float _stateExploratory;
			float _stateExploratoryPrev;

			float _activation;
			float _activationPrev;

			float _averageSurprise; // Use to keep track of importance for prediction. If current error is greater than average, then attention is > 0.5 else < 0.5 (sigmoid)

			PredictionNode()
				: _state(0.0f), _statePrev(0.0f), _stateExploratory(0.0f), _stateExploratoryPrev(0.0f),
				_activation(0.0f), _activationPrev(0.0f), _averageSurprise(0.0f)
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

		std::vector<int> _qInputIndices;
		std::vector<float> _qInputOffsets;

		std::vector<int> _actionInputIndices;

		float _prevValue;

	public:
		float _stateLeak;
		float _exploratoryNoise;
		float _exploratoryNoiseInput;
		float _gamma;
		float _gammaLambda;
		float _actionRandomizeChance;
		float _qAlpha;
		float _learnInputFeedBackPred;
		float _learnInputFeedBackRL;

		IPRSDRRL()
			: _prevValue(0.0f),
			_stateLeak(1.0f),
			_exploratoryNoise(0.8f),
			_exploratoryNoiseInput(0.5f),
			_gamma(0.99f),
			_gammaLambda(0.98f),
			_actionRandomizeChance(0.01f),
			_qAlpha(0.5f),
			_learnInputFeedBackPred(0.005f),
			_learnInputFeedBackRL(0.005f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator, bool learn = true);

		void setState(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value * _stateLeak + (1.0f - _stateLeak) * getAction(index));
		}

		void setState(int x, int y, float value) {
			setState(x + y * _layers.front()._sdr.getVisibleWidth(), value);
		}

		float getAction(int index) const {
			return _inputPredictionNodes[index]._stateExploratory;
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