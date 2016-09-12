cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(icptool)

find_package(PCL 1.5 REQUIRED COMPONENTS common registration kdtree)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (icptool icptool.cpp)
target_link_libraries (icptool ${PCL_LIBRARIES})