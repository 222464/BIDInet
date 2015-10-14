#pragma once

#include <vector>
#include <random>

namespace convnet {
	class Map {
	private:
		int _width, _height;

		std::vector<float> _map;

	public:
		void create(int width, int height) {
			_width = width;
			_height = height;

			_map.resize(_width * _height);
		}

		float operator[](int index) const {
			return _map[index];
		}

		float &operator[](int index) {
			return _map[index];
		}

		float atXY(int x, int y) const {
			return _map[x + y * _width];
		}

		float &atXY(int x, int y) {
			return _map[x + y * _width];
		}

		int getWidth() const {
			return _width;
		}

		int getHeight() const {
			return _height;
		}

		void clear(float value) {
			int size = _map.size();

			_map.clear();
			_map.assign(size, value);
		}
	};
}