#pragma once

#include "../sdr/IRSDR.h"
#include "SDRRL.h"

#include <assert.h>

namespace deep {
	class CSRL {
	public:
		enum InputType {
			_state, _action
		};

		enum ActionType {
			_attention = 0, _learn, _reward, _numActionTypes
		};

		struct Connection {
			unsigned short _index;

			float _weight;

			float _trace;

			Connection()
				: _trace(0.0f)
			{}
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveRadius, _recurrentRadius, _lateralRadius, _predictiveRadius, _feedBackRadius;

			int _numRecurrentInputs;

			float _learnFeedForward, _learnRecurrent, _learnLateral;

			float _learnFeedBackPred, _learnPredictionPred;
			float _learnFeedBackRL, _learnPredictionRL;
			float _drift;

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
			float _gateLateralAlpha;
			float _gateThresholdAlpha;
			int _gateSolveIter;
			float _qAlpha;
			float _actionAlpha, _actionDeriveAlpha;
			int _actionDeriveIterations;
			float _explorationStdDev;
			float _explorationBreak;
			float _epsilon;
					
			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(6), _recurrentRadius(5), _lateralRadius(4), _predictiveRadius(5), _feedBackRadius(6),
				_learnFeedForward(0.01f), _learnRecurrent(0.01f), _learnLateral(0.3f), _numRecurrentInputs(8),
				_learnFeedBackPred(0.05f), _learnPredictionPred(0.05f),
				_learnFeedBackRL(0.05f), _learnPredictionRL(0.05f),
				_drift(0.0f),
				_sdrIterSettle(17), _sdrIterMeasure(4), _sdrLeak(0.1f),
				_sdrStepSize(0.04f), _sdrLambda(0.95f), _sdrHiddenDecay(0.01f), _sdrWeightDecay(0.0001f),
				_sparsity(0.1f), _sdrLearnThreshold(0.01f), _sdrNoise(0.01f),
				_sdrBaselineDecay(0.01f), _sdrSensitivity(10.0f),
				_averageSurpriseDecay(0.01f),
				_surpriseLearnFactor(2.0f),
				_cellsPerColumn(16),
				_cellSparsity(0.1f),
				_gamma(0.99f),
				_gammaLambda(0.95f),
				_gateFeedForwardAlpha(0.05f),
				_gateLateralAlpha(0.1f),
				_gateThresholdAlpha(0.005f),
				_gateSolveIter(5),
				_qAlpha(0.01f),
				_actionAlpha(0.5f), _actionDeriveAlpha(0.05f), _actionDeriveIterations(30),
				_explorationStdDev(0.1f), _explorationBreak(0.01f), _epsilon(0.05f)
			{}
		};

		struct PredictionNode {
			std::vector<Connection> _feedBackConnections;
			std::vector<Connection> _predictiveConnections;

			Connection _bias;

			SDRRL _sdrrl;

			std::vector<float> _rewardInputs;

			float _localReward;

			float _baseline;

			float _state;
			float _statePrev;

			float _stateOutput;
			float _stateOutputPrev;

			PredictionNode()
				: _localReward(0.0f), _state(0.0f), _statePrev(0.0f), _stateOutput(0.0f), _stateOutputPrev(0.0f),
				_baseline(0.0f)
			{}
		};

		struct InputPredictionNode {
			std::vector<Connection> _feedBackConnections;

			Connection _bias;

			SDRRL _sdrrl;

			std::vector<float> _rewardInputs;

			float _localReward;

			float _baseline;

			float _state;
			float _statePrev;

			float _stateOutput;
			float _stateOutputPrev;

			InputPredictionNode()
				: _localReward(0.0f), _state(0.0f), _statePrev(0.0f), _stateOutput(0.0f), _stateOutputPrev(0.0f),
				_baseline(0.0f)
			{}
		};

		struct Layer {
			sdr::IRSDR _sdr;

			std::vector<PredictionNode> _predictionNodes;
		};

		struct QNode {
			int _index;
			float _offset;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		std::vector<InputPredictionNode> _inputPredictionNodes;

		std::vector<InputType> _inputTypes;

		std::vector<float> _lastLayerRewardOffsets;

		float _prevValue;

	public:
		float _learnFeedBackPred;
		float _learnFeedBackRL;
		float _drift;

		float _averageSurpriseDecay;
		float _surpriseLearnFactor;

		int _numRecurrentInputs;

		int _cellsPerColumn;
		float _cellSparsity;
		float _gamma;
		float _gammaLambda;
		float _gateFeedForwardAlpha;
		float _gateLateralAlpha;
		float _gateThresholdAlpha;
		int _gateSolveIter;
		float _qAlpha;
		float _actionAlpha, _actionDeriveAlpha;
		int _actionDeriveIterations;
		float _explorationStdDev;
		float _explorationBreak;
		float _epsilon;

		float _sdrBaselineDecay;
		float _sdrSensitivity;
		int _sdrIterSettle;
		int _sdrIterMeasure;
		float _sdrLeak;

		CSRL()
			: _learnFeedBackPred(0.05f),
			_learnFeedBackRL(0.05f),
			_drift(0.0f),
			_averageSurpriseDecay(0.01f),
			_surpriseLearnFactor(2.0f),
			_numRecurrentInputs(8),
			_cellsPerColumn(16),
			_cellSparsity(0.125f),
			_gamma(0.99f),
			_gammaLambda(0.95f),
			_gateFeedForwardAlpha(0.05f),
			_gateLateralAlpha(0.1f),
			_gateThresholdAlpha(0.005f),
			_gateSolveIter(5),
			_qAlpha(0.01f),
			_actionAlpha(0.5f), _actionDeriveAlpha(0.05f), _actionDeriveIterations(30),
			_explorationStdDev(0.1f), _explorationBreak(0.01f), _epsilon(0.05f),
			_sdrBaselineDecay(0.01f),
			_sdrSensitivity(10.0f),
			_sdrIterSettle(17),
			_sdrIterMeasure(4),
			_sdrLeak(0.1f),
			_prevValue(0.0f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator, bool learn = true);

		void setInput(int index, float value) {
			assert(_inputTypes[index] == _state);

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