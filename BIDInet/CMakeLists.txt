cmake_minimum_required(VERSION 2.8)

project(BIDInet)

include_directories("${PROJECT_SOURCE_DIR}/source")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
find_package(OpenCL REQUIRED)
 
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(SFML 2 REQUIRED system window graphics)
 
include_directories(${SFML_INCLUDE_DIR})
 
find_package(Box2D REQUIRED)
 
include_directories(${BOX2D_INCLUDE_DIRS})

add_executable(BIDInet "${PROJECT_SOURCE_DIR}/source/Main.cpp")

target_link_libraries(BIDInet ${OpenCL_LIBRARIES})
target_link_libraries(BIDInet ${SFML_LIBRARIES})
target_link_libraries(BIDInet ${BOX2D_LIBRARIES})