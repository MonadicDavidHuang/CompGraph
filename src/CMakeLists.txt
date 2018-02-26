cmake_minimum_required(VERSION 3.13)

project(ComputationGraph)

add_definitions("-Wall -std=c++11") # <= new

add_executable(Main
        implementation.cpp
        mnistLoader.h
        )

file(COPY ../Data/data.txt DESTINATION "${CMAKE_BINARY_DIR}")
