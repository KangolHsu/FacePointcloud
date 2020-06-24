cmake_minimum_required(VERSION 2.8.12)  # Ships with Ubuntu 14.04 (Trusty)

add_definitions( -D_CRT_SECURE_NO_WARNINGS )

find_package(PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )


set(OpenCV_DIR D:/3rdParty/OpenCV/opencv3.2.0/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/utils)


add_executable( SampleFacePC SampleFacePC.cpp ${PROJECT_SOURCE_DIR}/utils/ini.c)
target_link_libraries( SampleFacePC ${PCL_LIBRARIES} ${OpenCV_LIBS} )