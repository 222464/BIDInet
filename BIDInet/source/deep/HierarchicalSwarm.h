#pragma once

#include "SelfOptimizingUnit.h"

#include <array>

namespace deep {
	class HierarchicalSwarm {
	public:
		struct Column {
			SelfOptimizingUnit _sou; // 2 outputs: state for next layer, action/prediction

			std::array<float, 3> _prevStates;

			Column()
			{
				_prevStates.assign(0.0f);
			}
		};

		struct LayerDesc {
			int _width, _height;
			int _cellsPerColumn;
			int _recurrentActions;

			int _ffRadius, _lRadius, _fbRadius;

			float _ffAlpha, _inhibAlpha, _biasAlpha;

			float _qAlpha;
			float _actionAlpha;

			float _expPert;
			float _expBreak;

			float _gamma, _lambdaGamma;

			float _cellSparsity;

			LayerDesc()
				: _width(16), _height(16),
				_cellsPerColumn(32),
				_recurrentActions(4),
				_ffRadius(1), _lRadius(1), _fbRadius(1),
				_ffAlpha(0.01f), _inhibAlpha(0.1f), _biasAlpha(0.005f),
				_qAlpha(0.01f),
				_actionAlpha(0.1f),
				_expPert(0.01f),
				_expBreak(0.01f),
				_gamma(0.99f), _lambdaGamma(0.98f),
				_cellSparsity(0.1f)
			{}
		};

		struct Layer {
			std::vector<Column> _columns;
		};

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

	public:
		void createRandom(const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, std::mt19937 &generator);

		void setInput(int index, float value) {
			_layers.front()._columns[index]._sou.setState(0, value);
		}

		float getAction(int index) const {
			return _layers.front()._columns[index]._sou.getAction(2);
		}

		void simStep(int subIter, float reward, std::mt19937 &generator);

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayes() const {
			return _layers;
		}
	};
}