#include "CSRLVisualizer.h"

using namespace vis;

void CSRLVisualizer::create(unsigned int width) {
	_rt.create(width, width, false);
}

void CSRLVisualizer::update(sf::RenderTexture &target, const sf::Vector2f &position, const sf::Vector2f &scale, const deep::CSRL &csrl, int seed) {
	std::mt19937 generator(seed);
	
	std::vector<std::shared_ptr<sf::Image>> images;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	int maxWidth = 0;
	int maxHeight = 0;

	for (int l = 0; l < csrl.getLayerDescs().size(); l++) {
		sf::Color sheathColorPositive = sf::Color::White;

		sheathColorPositive.r = dist01(generator) * 255.0f;
		sheathColorPositive.g = dist01(generator) * 255.0f;
		sheathColorPositive.b = dist01(generator) * 255.0f;

		sheathColorPositive.a = 210;

		sf::Color sheathColorNegative = sf::Color::White;

		sheathColorNegative.r = dist01(generator) * 255.0f;
		sheathColorNegative.g = dist01(generator) * 255.0f;
		sheathColorNegative.b = dist01(generator) * 255.0f;

		sheathColorNegative.a = 210;

		sf::Color cellColor = sf::Color::White;

		cellColor.r = dist01(generator) * 255.0f;
		cellColor.g = dist01(generator) * 255.0f;
		cellColor.b = dist01(generator) * 255.0f;

		cellColor.a = 210;

		for (int c = 0; c < csrl.getLayerDescs()[l]._cellsPerColumn; c++) {
			std::shared_ptr<sf::Image> img = std::make_shared<sf::Image>();

			images.push_back(img);

			img->create(csrl.getLayerDescs()[l]._width * 3, csrl.getLayerDescs()[l]._height * 3);

			maxWidth = std::max<int>(img->getSize().x, maxWidth);
			maxHeight = std::max<int>(img->getSize().y, maxHeight);

			for (int x = 0; x < csrl.getLayerDescs()[l]._width; x++)
				for (int y = 0; y < csrl.getLayerDescs()[l]._height; y++) {
					int index = x + y * csrl.getLayerDescs()[l]._width;

					float s = csrl.getLayers()[l]._sdr.getHiddenState(index);

					sf::Color sheathColor = sf::Color::White;

					sheathColor.r = s * sheathColorPositive.r + (1.0f - s) * sheathColorNegative.r;
					sheathColor.g = s * sheathColorPositive.g + (1.0f - s) * sheathColorNegative.g;
					sheathColor.b = s * sheathColorPositive.b + (1.0f - s) * sheathColorNegative.b;

					sheathColor.a = sheathColorPositive.a * s;

					for (int dx = 0; dx < 3; dx++)
						for (int dy = 0; dy < 3; dy++) {
							img->setPixel(x * 3 + dx, y * 3 + dy, sheathColor);
						}

					sf::Color thisCellColor = cellColor;

					thisCellColor.a *= csrl.getLayers()[l]._predictionNodes[index]._sdrrl.getCellState(c);

					img->setPixel(x * 3 + 1, y * 3 + 1, thisCellColor);
				}
		}
	}

	const float heightStep = 1.5f;
	const float transparency = 0.3f;
	const int cellLayerSteps = 3;

	int h = 0;

	sf::Texture imageTexture;

	for (int i = 0; i < images.size(); i++) {
		// Render to RT
		_rt.setActive();

		imageTexture.loadFromImage(*images[i]);

		imageTexture.setSmooth(false);
		
		sf::Sprite imageSprite;
		imageSprite.setTexture(imageTexture);

		imageSprite.setOrigin(imageTexture.getSize().x * 0.5f, imageTexture.getSize().y * 0.5f);

		imageSprite.setRotation(45.0f);
		imageSprite.setPosition(_rt.getSize().x * 0.5f, _rt.getSize().y * 0.5f);
		imageSprite.setScale(static_cast<float>(_rt.getSize().x) / maxWidth * 0.75f, static_cast<float>(_rt.getSize().y) / maxHeight * 0.75f);

		sf::RenderStates clearStates;
		clearStates.blendMode = sf::BlendNone;

		sf::RectangleShape clearShape;
		clearShape.setSize(sf::Vector2f(_rt.getSize().x, _rt.getSize().y));
		clearShape.setFillColor(sf::Color::Transparent);

		_rt.draw(clearShape, clearStates);

		_rt.draw(imageSprite);

		_rt.display();

		// Render rt to main image
		target.setActive();

		sf::Sprite transformedSprite;
		transformedSprite.setTexture(_rt.getTexture());
		transformedSprite.setOrigin(transformedSprite.getTexture()->getSize().x * 0.5f, transformedSprite.getTexture()->getSize().y * 0.5f);
	
		transformedSprite.setScale(scale.x * 0.5f, scale.y * 0.25f);
		transformedSprite.setColor(sf::Color(255, 255, 255, 255.0f * transparency));

		target.setSmooth(true);

		for (int s = 0; s < cellLayerSteps; s++) {
			transformedSprite.setPosition(position.x, position.y - h * heightStep);
			target.draw(transformedSprite);

			h++;
		}
	}

	target.display();
}