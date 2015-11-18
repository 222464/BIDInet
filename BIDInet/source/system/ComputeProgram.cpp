#include <system/ComputeProgram.h>

#include <fstream>
#include <iostream>

using namespace d3d;

bool ComputeProgram::createAsset(const std::string &name, void* pData) {
	// pData is compute system
	ComputeSystem* pComputeSystem = static_cast<ComputeSystem*>(pData);

	std::ifstream fromFile(name);

	if (!fromFile.is_open()) {
		std::cerr << "Could not open file " << name << "!" << std::endl;
		return false;
	}

	std::string source = "";

	while (!fromFile.eof() && fromFile.good()) {
		std::string line;

		std::getline(fromFile, line);

		source += line + "\n";
	}

	_program = cl::Program(pComputeSystem->getContext(), source);

	if (_program.build(std::vector<cl::Device>(1, pComputeSystem->getDevice())) != CL_SUCCESS) {
		std::cerr << "Error building: " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(pComputeSystem->getDevice()) << std::endl;
		return false;
	}

	return true;
}