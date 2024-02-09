#!/bin/bash 

rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --config release
cp cv_yolo_neuromatrix ../../../bin
cd ..


