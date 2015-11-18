#pragma once

#include <system/Uncopyable.h>
#include <CL/cl.hpp>
#include <SFML/Window.hpp>

namespace d3d {
	class ComputeSystem : public Uncopyable {
	public:
		enum DeviceType {
			_cpu, _gpu, _all, _none
		};

	private:
		cl::Platform _platform;
		cl::Device _device;
		cl::Context _context;
		cl::CommandQueue _queue;

	public:
		bool create(DeviceType type, bool createFromGLContext);

		cl::Platform &getPlatform() {
			return _platform;
		}

		cl::Device &getDevice() {
			return _device;
		}

		cl::Context &getContext() {
			return _context;
		}

		cl::CommandQueue &getQueue() {
			return _queue;
		}
	};
}