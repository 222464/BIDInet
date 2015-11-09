#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_VIDEO_TEST

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <random>

#include <sdr/IPredictiveRSDR.h>

#include <dirent.h>

using namespace cv;

int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 800), "BIDInet", sf::Style::Default);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	std::string fileNameRoot = "C:/Users/Eric/Downloads/hollywood.tar/hollywood/hollywood/videoclips/";

	const int frameSkip = 2;
	const float videoScale = 0.45f;

	DIR *dir;
	dirent *ent;

	std::vector<std::string> fileNames;

	if ((dir = opendir(fileNameRoot.c_str())) != nullptr) {
		while ((ent = readdir(dir)) != nullptr) {
			fileNames.push_back(ent->d_name);
		}

		closedir(dir);
	}

	sf::RenderTexture rescaleRT;
	rescaleRT.create(128, 128);

	std::vector<sdr::IPredictiveRSDR::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 32;
	layerDescs[0]._height = 32;

	layerDescs[1]._width = 24;
	layerDescs[1]._height = 24;

	layerDescs[2]._width = 16;
	layerDescs[2]._height = 16;

	sdr::IPredictiveRSDR prsdr;

	prsdr.createRandom(rescaleRT.getSize().x, rescaleRT.getSize().y, 16, layerDescs, -0.01f, 0.01f, 0.01f, 0.05f, 0.1f, generator);

	// Train for a bit
	std::uniform_int_distribution<int> fileDist(0, fileNames.size() - 1);
	
	for (int iter = 0; iter < 5; iter++) {
		int index = fileDist(generator);

		std::string fullName = fileNameRoot + fileNames[index];

		VideoCapture capture(fullName);
		Mat frame;

		if (!capture.isOpened())
			std::cerr << "Could not open capture: " << fullName << std::endl;

		std::cout << "Running through capture: " << fileNames[index] << std::endl;

		do {
			for (int i = 0; i < frameSkip; i++) {
				capture >> frame;

				if (frame.empty())
					break;
			}

			if (frame.empty())
				break;

			sf::Image img;

			img.create(frame.cols, frame.rows);

			for (int x = 0; x < img.getSize().x; x++)
				for (int y = 0; y < img.getSize().y; y++) {
					sf::Uint8 r = frame.data[(x + y * img.getSize().x) * 3 + 0];
					sf::Uint8 g = frame.data[(x + y * img.getSize().x) * 3 + 1];
					sf::Uint8 b = frame.data[(x + y * img.getSize().x) * 3 + 2];

					img.setPixel(x, y, sf::Color(r, g, b));
				}

			sf::Texture tex;
			tex.loadFromImage(img);

			tex.setSmooth(true);

			sf::Sprite s;
			
			s.setPosition(rescaleRT.getSize().x * 0.5f, rescaleRT.getSize().y * 0.5f);

			s.setTexture(tex);

			s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));

			s.setScale(sf::Vector2f(videoScale, videoScale));

			rescaleRT.clear();

			rescaleRT.draw(s);

			rescaleRT.display();

			sf::Image reImg = rescaleRT.getTexture().copyToImage();

			for (int x = 0; x < reImg.getSize().x; x++)
				for (int y = 0; y < reImg.getSize().y; y++) {
					sf::Color c = reImg.getPixel(x, y);

					float mono = (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f) * 0.3333f;

					prsdr.setInput(x, y, mono);
				}

			prsdr.simStep(generator);

			std::cout << "f";

		} while (!frame.empty());

		std::cout << "Iteration " << iter << std::endl;
	}

	// ---------------------------- Game Loop -----------------------------
	
	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	std::uniform_real_distribution<float> noise(0.0f, 1.0f);

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

		window.clear();

		for (int x = 0; x < rescaleRT.getSize().x; x++)
			for (int y = 0; y < rescaleRT.getSize().y; y++) {
				prsdr.setInput(x, y, prsdr.getPrediction(x, y));
			}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T))
			prsdr.simStep(generator, false);

		// Display prediction
		sf::Image img;
		
		img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);
		
		for (int x = 0; x < rescaleRT.getSize().x; x++)
			for (int y = 0; y < rescaleRT.getSize().y; y++) {
				sf::Color c;

				c.r = c.g = c.b = 255.0f * std::min(1.0f, std::max(0.0f, prsdr.getPrediction(x, y)));

				img.setPixel(x, y, c);
			}

		sf::Texture tex;

		tex.loadFromImage(img);

		sf::Sprite s;

		s.setTexture(tex);

		s.setScale(sf::Vector2f(6.0f, 6.0f));

		window.draw(s);

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

	} while (!quit);

	return 0;
}

#endif