#pragma once

#include <SFML/Graphics.hpp>
#include <deep/CSRL.h>
#include <memory>
#include <algorithm>

namespace vis {
	class CSRLVisualizer {
	private:
		sf::RenderTexture _rt;
	public:
		void create(unsigned int width);

		void update(sf::RenderTexture &target, const sf::Vector2f &position, const sf::Vector2f &scale, const deep::CSRL &csrl, int seed);
	};
}