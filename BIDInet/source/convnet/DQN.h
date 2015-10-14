#pragma once

#include "ConvNet.h"
#include <list>

namespace convnet {
	class DQN {
	public:
		struct ReplaySample {
			std::vector<Map> _inputs;

			std::vector<float> _actions;
			std::vector<float> _originalActions;

			float _q;
			float _originalQ;
		};

	private:
		std::list<ReplaySample> _replaySamples;
		
		float _prevValue;

		std::vector<float> _exploratoryActions;
		std::vector<float> _actions;

		std::vector<Map> _inputMapsPrev;

	public:
		ConvNet _net;

		float _qAlpha;
		float _qGamma;
		float _actionPerturbationStdDev;
		float _actionBreakChance;

		int _maxReplayChainSize;
		int _replayIterations;

		DQN()
			: _prevValue(0.0f),
			_qAlpha(0.5f),
			_qGamma(0.99f),
			_actionPerturbationStdDev(0.05f),
			_actionBreakChance(0.01f),
			_maxReplayChainSize(512),
			_replayIterations(64)
		{}

		void simStep(float reward, std::mt19937 &generator);

		float getExploratoryAction(int index) const {
			return _exploratoryActions[index];
		}
	};
}