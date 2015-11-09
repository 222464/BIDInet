#include "IRSDR.h"

#include <algorithm>

#include <SFML/System.hpp>
#include <SFML/Window.hpp>

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sdr;

void IRSDR::createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int recurrentRadius, int lateralRadius, float initMinWeight, float initMaxWeight, float initMinInhibition, float initMaxInhibition, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	std::uniform_real_distribution<float> inhibitionDist(initMinInhibition, initMaxInhibition);

	_visibleWidth = visibleWidth;
	_visibleHeight = visibleHeight;
	_hiddenWidth = hiddenWidth;
	_hiddenHeight = hiddenHeight;

	_receptiveRadius = receptiveRadius;
	_recurrentRadius = recurrentRadius;

	int numVisible = visibleWidth * visibleHeight;
	int numHidden = hiddenWidth * hiddenHeight;
	int receptiveSize = std::pow(receptiveRadius * 2 + 1, 2);
	int recurrentSize = std::pow(recurrentRadius * 2 + 1, 2);
	int lateralSize = std::pow(lateralRadius * 2 + 1, 2);

	_visible.resize(numVisible);

	_hidden.resize(numHidden);

	float hiddenToVisibleWidth = static_cast<float>(visibleWidth) / static_cast<float>(hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(visibleHeight) / static_cast<float>(hiddenHeight);

	for (int hi = 0; hi < numHidden; hi++) {
		int hx = hi % hiddenWidth;
		int hy = hi / hiddenWidth;

		int centerX = std::round(hx * hiddenToVisibleWidth);
		int centerY = std::round(hy * hiddenToVisibleHeight);

		_hidden[hi]._threshold = initThreshold;

		// Receptive
		_hidden[hi]._feedForwardConnections.reserve(receptiveSize);

		for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
			for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
				int vx = centerX + dx;
				int vy = centerY + dy;

				if (vx >= 0 && vx < visibleWidth && vy >= 0 && vy < visibleHeight) {
					int vi = vx + vy * visibleWidth;

					Connection c;

					c._weight = weightDist(generator);
					c._index = vi;

					_hidden[hi]._feedForwardConnections.push_back(c);
				}
			}

		_hidden[hi]._feedForwardConnections.shrink_to_fit();

		// Recurrent
		if (recurrentRadius != -1) {
			_hidden[hi]._recurrentConnections.reserve(recurrentSize);

			for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
				for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
					if (dx == 0 && dy == 0)
						continue;

					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenHeight) {
						int hio = hox + hoy * hiddenWidth;

						Connection c;

						c._weight = weightDist(generator);
						c._index = hio;

						_hidden[hi]._recurrentConnections.push_back(c);
					}
				}

			_hidden[hi]._recurrentConnections.shrink_to_fit();
		}

		_hidden[hi]._lateralConnections.reserve(recurrentSize);

		for (int dx = -lateralRadius; dx <= lateralRadius; dx++)
			for (int dy = -lateralRadius; dy <= lateralRadius; dy++) {
				if (dx == 0 && dy == 0)
					continue;

				int hox = hx + dx;
				int hoy = hy + dy;

				if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenHeight) {
					int hio = hox + hoy * hiddenWidth;

					Connection c;

					c._weight = inhibitionDist(generator);
					c._index = hio;

					_hidden[hi]._lateralConnections.push_back(c);
				}
			}

		_hidden[hi]._lateralConnections.shrink_to_fit();
	}
}

void IRSDR::activate(int settleIter, int measureIter, float leak, float noise, std::mt19937 &generator) {
	std::normal_distribution<float> noiseDist(0.0f, noise);

	std::vector<float> visibleErrors(_visible.size());
	std::vector<float> hiddenErrors(_hidden.size());

	for (int hi = 0; hi < _hidden.size(); hi++) {
		_hidden[hi]._activation = 0.0f;

		_hidden[hi]._state = 0.0f;
	}

	for (int it = 0; it < settleIter; it++) {
		for (int vi = 0; vi < _visible.size(); vi++)
			visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

		for (int hi = 0; hi < _hidden.size(); hi++)
			hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

		for (int hi = 0; hi < _hidden.size(); hi++) {
			float excitation = 0.0f;

			for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
				excitation += visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index] * _hidden[hi]._feedForwardConnections[ci]._weight;

			for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
				excitation += hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index] * _hidden[hi]._recurrentConnections[ci]._weight;

			float inhibition = 0.0f;

			for (int ci = 0; ci < _hidden[hi]._lateralConnections.size(); ci++)
				inhibition += _hidden[_hidden[hi]._lateralConnections[ci]._index]._spikePrev * _hidden[hi]._lateralConnections[ci]._weight;

			_hidden[hi]._activation = (1.0f - leak) * _hidden[hi]._activation + excitation - inhibition;

			if (_hidden[hi]._activation > _hidden[hi]._threshold) {
				_hidden[hi]._activation = 0.0f;
				_hidden[hi]._spike = 1.0f;
			}
			else
				_hidden[hi]._spike = 0.0f;
		}

		for (int hi = 0; hi < _hidden.size(); hi++)
			_hidden[hi]._spikePrev = _hidden[hi]._spike;

		reconstructFromSpikes();
	}

	float measureIterInv = 1.0f / measureIter;

	for (int it = 0; it < measureIter; it++) {
		for (int vi = 0; vi < _visible.size(); vi++)
			visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

		for (int hi = 0; hi < _hidden.size(); hi++)
			hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

		for (int hi = 0; hi < _hidden.size(); hi++) {
			float excitation = 0.0f;

			for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
				excitation += visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index] * _hidden[hi]._feedForwardConnections[ci]._weight;

			for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
				excitation += hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index] * _hidden[hi]._recurrentConnections[ci]._weight;

			float inhibition = 0.0f;

			for (int ci = 0; ci < _hidden[hi]._lateralConnections.size(); ci++)
				inhibition += _hidden[_hidden[hi]._lateralConnections[ci]._index]._spikePrev * _hidden[hi]._lateralConnections[ci]._weight;

			_hidden[hi]._activation = (1.0f - leak) * _hidden[hi]._activation + excitation - inhibition;

			if (_hidden[hi]._activation > _hidden[hi]._threshold) {
				_hidden[hi]._activation = 0.0f;
				_hidden[hi]._spike = 1.0f;
			}
			else
				_hidden[hi]._spike = 0.0f;

			_hidden[hi]._state += measureIterInv * _hidden[hi]._spike;
		}

		for (int hi = 0; hi < _hidden.size(); hi++)
			_hidden[hi]._spikePrev = _hidden[hi]._spike;

		reconstructFromSpikes();
	}

	reconstructFromStates();
}

void IRSDR::reconstructFromSpikes() {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			_visible[_hidden[hi]._feedForwardConnections[ci]._index]._reconstruction += _hidden[hi]._feedForwardConnections[ci]._weight * _hidden[hi]._spike;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			_hidden[_hidden[hi]._recurrentConnections[ci]._index]._reconstruction += _hidden[hi]._recurrentConnections[ci]._weight * _hidden[hi]._spike;
	}
}

void IRSDR::reconstructFromStates() {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			_visible[_hidden[hi]._feedForwardConnections[ci]._index]._reconstruction += _hidden[hi]._feedForwardConnections[ci]._weight * _hidden[hi]._state;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			_hidden[_hidden[hi]._recurrentConnections[ci]._index]._reconstruction += _hidden[hi]._recurrentConnections[ci]._weight * _hidden[hi]._state;
	}
}

void IRSDR::reconstruct(const std::vector<float> &states, std::vector<float> &reconHidden, std::vector<float> &reconVisible) {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	reconVisible.clear();
	reconVisible.assign(_visible.size(), 0.0f);

	reconHidden.clear();
	reconHidden.assign(_hidden.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			reconVisible[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._feedForwardConnections[ci]._weight * states[hi];

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			reconHidden[_hidden[hi]._recurrentConnections[ci]._index] += _hidden[hi]._recurrentConnections[ci]._weight * states[hi];
	}
}

void IRSDR::reconstructFeedForward(const std::vector<float> &states, std::vector<float> &recon) {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);

	recon.clear();
	recon.assign(_visible.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			recon[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._feedForwardConnections[ci]._weight * states[hi];
	}
}

void IRSDR::learn(float learnFeedForward, float learnRecurrent, float learnLateral, float learnThreshold, float sparsity, float weightDecay, float maxWeightDelta) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float learn = _hidden[hi]._state;

		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			float delta = learnFeedForward * learn * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index] - weightDecay * _hidden[hi]._feedForwardConnections[ci]._weight;

			_hidden[hi]._feedForwardConnections[ci]._weight += std::min(maxWeightDelta, std::max(-maxWeightDelta, delta));
		}

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++) {
			float delta = learnRecurrent * learn * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index] - weightDecay * _hidden[hi]._recurrentConnections[ci]._weight;

			_hidden[hi]._recurrentConnections[ci]._weight += std::min(maxWeightDelta, std::max(-maxWeightDelta, delta));
		}

		for (int ci = 0; ci < _hidden[hi]._lateralConnections.size(); ci++)
			_hidden[hi]._lateralConnections[ci]._weight = std::max(0.0f, _hidden[hi]._lateralConnections[ci]._weight + learnLateral * (_hidden[hi]._state * _hidden[_hidden[hi]._lateralConnections[ci]._index]._state - sparsity * sparsity));


		_hidden[hi]._threshold = std::max(0.0f, _hidden[hi]._threshold + (_hidden[hi]._state - sparsity) * learnThreshold);
	}
}

void IRSDR::learn(const std::vector<float> &rewards, float lambda, float learnFeedForward, float learnRecurrent, float learnLateral, float learnThreshold, float sparsity, float weightDecay, float maxWeightDelta) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float learn = _hidden[hi]._state;

		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			float delta = learnFeedForward * rewards[hi] * _hidden[hi]._feedForwardConnections[ci]._trace - weightDecay * _hidden[hi]._feedForwardConnections[ci]._weight;

			_hidden[hi]._feedForwardConnections[ci]._weight += std::min(maxWeightDelta, std::max(-maxWeightDelta, delta));

			_hidden[hi]._feedForwardConnections[ci]._trace = lambda * _hidden[hi]._feedForwardConnections[ci]._trace + learn * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index];
		}

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++) {
			float delta = learnRecurrent * rewards[hi] * _hidden[hi]._recurrentConnections[ci]._trace - weightDecay * _hidden[hi]._recurrentConnections[ci]._weight;

			_hidden[hi]._recurrentConnections[ci]._weight += std::min(maxWeightDelta, std::max(-maxWeightDelta, delta));

			_hidden[hi]._recurrentConnections[ci]._trace = lambda * _hidden[hi]._recurrentConnections[ci]._trace + learn * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index];
		}

		for (int ci = 0; ci < _hidden[hi]._lateralConnections.size(); ci++)
			_hidden[hi]._lateralConnections[ci]._weight = std::max(0.0f, _hidden[hi]._lateralConnections[ci]._weight + learnLateral * (_hidden[hi]._state * _hidden[_hidden[hi]._lateralConnections[ci]._index]._state - sparsity * sparsity));

		_hidden[hi]._threshold = std::max(0.0f, _hidden[hi]._threshold + (_hidden[hi]._state - sparsity) * learnThreshold);
	}
}

void IRSDR::getVHWeights(int hx, int hy, std::vector<float> &rectangle) const {
	float hiddenToVisibleWidth = static_cast<float>(_visibleWidth) / static_cast<float>(_hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(_visibleHeight) / static_cast<float>(_hiddenHeight);

	int dim = _receptiveRadius * 2 + 1;

	rectangle.resize(dim * dim, 0.0f);

	int hi = hx + hy * _hiddenWidth;

	int centerX = std::round(hx * hiddenToVisibleWidth);
	int centerY = std::round(hy * hiddenToVisibleHeight);

	for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
		int index = _hidden[hi]._feedForwardConnections[ci]._index;

		int vx = index % _visibleWidth;
		int vy = index / _visibleWidth;

		int dx = vx - centerX;
		int dy = vy - centerY;

		int rx = dx + _receptiveRadius;
		int ry = dy + _receptiveRadius;

		rectangle[rx + ry * dim] = _hidden[hi]._feedForwardConnections[ci]._weight;
	}
}

void IRSDR::stepEnd() {
	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._statePrev = _hidden[hi]._state;
}