#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_RUNNER

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <runner/Runner.h>

#include <bidinet/BIDInet.h>

#include <deep/SDRRL.h>

#include <time.h>
#include <iostream>
#include <random>

#include <deep/FERL.h>
#include <deep/SFERL.h>

int main() {
	sf::RenderWindow window;

	sf::ContextSettings glContextSettings;
	glContextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(800, 600), "BIDInet", sf::Style::Default, glContextSettings);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	std::mt19937 generator(time(nullptr));

	/*sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram program;

	program.loadFromFile("resources/bidinet.cl", cs);

	bidi::BIDInet bidinet;

	std::vector<bidi::BIDInet::InputType> inputTypes(64, bidi::BIDInet::_state);

	const int numStates = 3 + 3 + 2 + 2 + 1 + 2 + 2;
	const int numActions = 3 + 3 + 2 + 2;
	const int numQ = 8;

	for (int i = 0; i < numStates; i++)
		inputTypes[i] = bidi::BIDInet::_state;

	for (int i = 0; i < numActions; i++)
		inputTypes[numStates + i] = bidi::BIDInet::_action;

	std::vector<bidi::BIDInet::LayerDesc> layerDescs(2);

	layerDescs[0]._fbRadius = 16;
	layerDescs[1]._width = 8;
	layerDescs[1]._height = 8;

	bidinet.createRandom(cs, program, 8, 8, inputTypes, layerDescs, -0.1f, 0.1f, 0.001f, 1.0f, generator);*/

	// Physics
	std::shared_ptr<b2World> world = std::make_shared<b2World>(b2Vec2(0.0f, -9.81f));

	const float pixelsPerMeter = 256.0f;

	const float groundWidth = 5000.0f;
	const float groundHeight = 5.0f;

	// Create ground
	b2BodyDef groundBodyDef;
	groundBodyDef.position.Set(0.0f, 0.0f);

	b2Body* groundBody = world->CreateBody(&groundBodyDef);

	b2PolygonShape groundBox;
	groundBox.SetAsBox(groundWidth * 0.5f, groundHeight * 0.5f);

	groundBody->CreateFixture(&groundBox, 0.0f);

	sf::Texture skyTexture;

	skyTexture.loadFromFile("resources/background1.png");

	skyTexture.setSmooth(true);

	sf::Texture floorTexture;
	
	floorTexture.loadFromFile("resources/floor1.png");

	floorTexture.setRepeated(true);
	floorTexture.setSmooth(true);

	Runner runner0;

	runner0.createDefault(world, b2Vec2(0.0f, 2.762f), 0.0f, 1);

	Runner runner1;

	runner1.createDefault(world, b2Vec2(0.0f, 2.762f), 0.0f, 2);

	deep::FERL ferl;

	int recCount = 4;
	int clockCount = 4;

	ferl.createRandom(3 + 3 + 2 + 2 + 1 + 2 + 2 + recCount + clockCount, 3 + 3 + 2 + 2 + recCount, 32, 0.01f, generator);

	std::vector<float> prevAction(ferl.getNumAction(), 0.0f);

	deep::SDRRL sdrrl;

	sdrrl.createRandom(3 + 3 + 2 + 2 + 1 + 2 + 2 + recCount + clockCount, 3 + 3 + 2 + 2 + recCount, 64, -0.1f, 0.1f, 0.01f, 1.0f, generator);

	std::vector<float> sprevAction(sdrrl.getNumActions(), 0.0f);

	// ---------------------------- Game Loop -----------------------------

	sf::View view = window.getDefaultView();

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

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

		//bidinet.simStep(cs, 0.0f, 0.98f, 0.001f, 0.95f, 0.01f, 0.01f, generator);

		const float maxRunnerBodyAngle = 0.3f;
		const float runnerBodyAngleStab = 10.0f;

		{
			float reward;
			
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::K))
				reward = -runner0._pBody->GetLinearVelocity().x;
			else
				reward = runner0._pBody->GetLinearVelocity().x;

			std::vector<float> state;

			runner0.getStateVector(state);

			std::vector<float> action(3 + 3 + 2 + 2 + recCount);

			for (int a = 0; a < recCount; a++)
				state.push_back(sprevAction[sprevAction.size() - recCount + a]);

			for (int a = 0; a < clockCount; a++)
				state.push_back(std::sin(steps / 60.0f * 2.0f * a * 2.0f * 3.141596f));

			for (int i = 0; i < state.size(); i++)
				sdrrl.setState(i, state[i]);

			sdrrl.simStep(reward, 0.1f, 0.99f, 0.01f, 0.05f, 0.005f, 0.002f, 0.01f, 64, 100.0f, 0.05f, 0.98f, 0.08f, 0.005f, 0.01f, 4.0f, generator);

			for (int i = 0; i < action.size(); i++)
				action[i] = sdrrl.getAction(i) * 0.5f + 0.5f;

			sprevAction = action;

			runner0.motorUpdate(action, 12.0f);

			// Keep upright
			if (std::abs(runner0._pBody->GetAngle()) > maxRunnerBodyAngle)
				runner0._pBody->SetAngularVelocity(-runnerBodyAngleStab * runner0._pBody->GetAngle());
		}

		{
			float reward;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::K))
				reward = -runner1._pBody->GetLinearVelocity().x;
			else
				reward = runner1._pBody->GetLinearVelocity().x;

			std::vector<float> state;

			runner1.getStateVector(state);

			std::vector<float> action(3 + 3 + 2 + 2 + recCount);

			for (int a = 0; a < recCount; a++)
				state.push_back(prevAction[prevAction.size() - recCount + a]);

			for (int a = 0; a < clockCount; a++)
				state.push_back(std::sin(steps / 60.0f * 2.0f * a * 2.0f * 3.141596f));

			//ferl.step(state, action, reward, 0.5f, 0.99f, 0.98f, 0.05f, 16, 4, 0.05f, 0.01f, 0.05f, 600, 64, 0.01f, generator);

			for (int i = 0; i < action.size(); i++)
				action[i] = action[i] * 0.5f + 0.5f;

			prevAction = action;

			runner1.motorUpdate(action, 12.0f);

			// Keep upright
			if (std::abs(runner1._pBody->GetAngle()) > maxRunnerBodyAngle)
				runner1._pBody->SetAngularVelocity(-runnerBodyAngleStab * runner1._pBody->GetAngle());
		}

		int subSteps = 1;

		for (int ss = 0; ss < subSteps; ss++) {
			world->ClearForces();

			world->Step(1.0f / 60.0f / subSteps, 64, 64);
		}

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::T) || steps % 100 == 1) {
			// -------------------------------------------------------------------

			if (!sf::Keyboard::isKeyPressed(sf::Keyboard::B))
				view.setCenter(runner1._pBody->GetPosition().x * pixelsPerMeter, -runner1._pBody->GetPosition().y * pixelsPerMeter);
			else
				view.setCenter(runner0._pBody->GetPosition().x * pixelsPerMeter, -runner0._pBody->GetPosition().y * pixelsPerMeter);

			// Draw sky
			sf::Sprite skySprite;
			skySprite.setTexture(skyTexture);

			window.setView(window.getDefaultView());

			window.draw(skySprite);

			window.setView(view);

			sf::RectangleShape floorShape;
			floorShape.setSize(sf::Vector2f(groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter));
			floorShape.setTexture(&floorTexture);
			floorShape.setTextureRect(sf::IntRect(0, 0, groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter));

			floorShape.setOrigin(sf::Vector2f(groundWidth * pixelsPerMeter * 0.5f, groundHeight * pixelsPerMeter * 0.5f));

			window.draw(floorShape);

			runner1.renderDefault(window, sf::Color::Blue, pixelsPerMeter);
			runner0.renderDefault(window, sf::Color::Red, pixelsPerMeter);

			sf::Image img;
			img.create(sdrrl.getNumCells(), 1);

			for (int i = 0; i < sdrrl.getNumCells(); i++) {
				sf::Color c = sf::Color::Black;

				c.r = c.g = c.b = 255.0f * sdrrl.getCellState(i);

				img.setPixel(i, 0, c);
			}

			float scale = 4.0f;

			sf::Texture tex;
			tex.loadFromImage(img);

			sf::Sprite s;
			s.setTexture(tex);

			s.setScale(sf::Vector2f(scale, scale));

			s.setPosition(sf::Vector2f(0.0f, window.getSize().y - scale * img.getSize().y));

			window.setView(window.getDefaultView());

			window.draw(s);

			window.setView(view);

			window.display();
		}
		else {
			if (steps % 100 == 0)
				std::cout << "Steps: " << steps << " Distance: " << runner0._pBody->GetPosition().x << " | " << runner1._pBody->GetPosition().x << std::endl;
		}

		//dt = clock.getElapsedTime().asSeconds();

		steps++;

	} while (!quit);

	world->DestroyBody(groundBody);

	return 0;
}

#endif