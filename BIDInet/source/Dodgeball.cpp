#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_DODGEBALL

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <runner/Runner.h>

#include <bidinet/BIDInet.h>

#include <deep/CSRL.h>

#include <time.h>
#include <iostream>
#include <random>

#include <deep/FERL.h>

const float ballSpeed = 0.01f;
const float ballRadius = 0.13f;
const float agentRadius = 0.05f;
const float agentSpeed = 0.03f;
const float agentFieldRadius = 0.3f;
const float agentFieldOutlineWidth = 0.005f;

std::vector<sf::Vector2f> dodgeballPositions;
std::vector<sf::Vector2f> dodgeballVelocities;
sf::Vector2f agentPosition;

void renderScene(sf::RenderTarget &rt) {
	sf::Vector2f size = sf::Vector2f(rt.getSize().x, rt.getSize().y);

	sf::RectangleShape r;

	r.setFillColor(sf::Color::Transparent);
	r.setOutlineColor(sf::Color::Green);
	r.setOutlineThickness(size.x * agentFieldOutlineWidth);
	r.setSize(sf::Vector2f(agentFieldRadius * size.x * 2.0f, agentFieldRadius * size.y * 2.0f));

	r.setOrigin(r.getSize() * 0.5f);
	r.setPosition(0.5f * size.x, 0.5f * size.y);

	rt.draw(r);

	for (int i = 0; i < dodgeballPositions.size(); i++) {
		sf::CircleShape c;

		c.setRadius(1.0f);
		c.setOrigin(1.0f, 1.0f);
		c.setScale(ballRadius * size.x, ballRadius * size.y);
		c.setPosition(dodgeballPositions[i].x * size.x, rt.getSize().y - dodgeballPositions[i].y * size.y);
		c.setFillColor(sf::Color::Red);

		rt.draw(c);
	}

	{
		sf::CircleShape c;

		c.setRadius(1.0f);
		c.setOrigin(1.0f, 1.0f);
		c.setScale(agentRadius * size.x, agentRadius * size.y);
		c.setPosition(agentPosition.x * size.x, rt.getSize().y - agentPosition.y * size.y);
		c.setFillColor(sf::Color::Green);

		rt.draw(c);
	}
}

int main() {
	std::mt19937 generator(time(nullptr));

	dodgeballPositions.resize(5);
	dodgeballVelocities.resize(dodgeballPositions.size());

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (int i = 0; i < dodgeballPositions.size(); i++) {
		dodgeballPositions[i] = sf::Vector2f(dist01(generator), dist01(generator));

		dodgeballVelocities[i].x = dist01(generator) * 2.0f - 1.0f;
		dodgeballVelocities[i].y = dist01(generator) * 2.0f - 1.0f;

		dodgeballVelocities[i] = dodgeballVelocities[i] * (ballSpeed / std::sqrt(dodgeballVelocities[i].x * dodgeballVelocities[i].x + dodgeballVelocities[i].y * dodgeballVelocities[i].y));
	}

	agentPosition = sf::Vector2f(0.5f, 0.5f);

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 800), "BIDInet", sf::Style::Default);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);
	
	sf::RenderTexture visionRT;

	visionRT.create(32, 32);

	deep::CSRL swarm;

	std::vector<deep::CSRL::LayerDesc> layerDescs(4);

	layerDescs[0]._width = 32;
	layerDescs[0]._height = 32;

	layerDescs[1]._width = 24;
	layerDescs[1]._height = 24;

	layerDescs[2]._width = 16;
	layerDescs[2]._height = 16;

	layerDescs[3]._width = 8;
	layerDescs[3]._height = 8;

	swarm.createRandom(2, layerDescs, -0.01f, 0.01f, 0.01f, 1.0f, generator);

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float averageReward = 0.0f;
	const float averageRewardDecay = 0.003f;

	int steps = 0;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		visionRT.clear();

		renderScene(visionRT);

		visionRT.display();

		sf::Image img = visionRT.getTexture().copyToImage();

		for (int x = 0; x < img.getSize().x; x++)
			for (int y = 0; y < img.getSize().y; y++) {
				sf::Color c = img.getPixel(x, y);

				float valR = 0.0f;
				float valG = 0.0f;

				if (c.r > 0)
					valR = 1.0f;

				if (c.g > 0)
					valG = 1.0f;

				swarm.setState(x, y, 0, valR);
				swarm.setState(x, y, 1, valG);
			}

		float reward = 1.0f;

		for (int i = 0; i < dodgeballPositions.size(); i++) {
			sf::Vector2f delta = dodgeballPositions[i] - agentPosition;

			float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

			if (dist < (ballRadius + agentRadius)) {
				reward -= 1.0f;
			}
		}

		averageReward = (1.0f - averageRewardDecay) * averageReward + averageRewardDecay * reward;

		swarm.simStep(1, reward, generator);

		agentPosition.x += agentSpeed * (swarm.getAction(3, 24) * 2.0f - 1.0f);
		agentPosition.y += agentSpeed * (swarm.getAction(3, 48) * 2.0f - 1.0f);

		agentPosition.x = std::min(0.5f + agentFieldRadius - agentRadius, std::max(0.5f - agentFieldRadius + agentRadius, agentPosition.x));
		agentPosition.y = std::min(0.5f + agentFieldRadius - agentRadius, std::max(0.5f - agentFieldRadius + agentRadius, agentPosition.y));

		for (int i = 0; i < dodgeballPositions.size(); i++) {
			dodgeballPositions[i] += dodgeballVelocities[i];

			if (dodgeballPositions[i].x < 0.0f) {
				dodgeballPositions[i].x = 0.0f;
				dodgeballVelocities[i].x *= -1.0f;
			}
			else if (dodgeballPositions[i].x > 1.0f) {
				dodgeballPositions[i].x = 1.0f;
				dodgeballVelocities[i].x *= -1.0f;
			}

			if (dodgeballPositions[i].y < 0.0f) {
				dodgeballPositions[i].y = 0.0f;
				dodgeballVelocities[i].y *= -1.0f;
			}
			else if (dodgeballPositions[i].y > 1.0f) {
				dodgeballPositions[i].y = 1.0f;
				dodgeballVelocities[i].y *= -1.0f;
			}
		}

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			window.clear();

			renderScene(window);

			window.display();
		}
		
		if (steps % 100 == 0)
			std::cout << "Steps: " << steps << " Average Reward: " << averageReward << std::endl;

		//dt = clock.getElapsedTime().asSeconds();

		steps++;

	} while (!quit);

	return 0;
}

#endif