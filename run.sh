#!/bin/bash

echo "Running IO limited benchmark"
python video_reading_benchmarks/main.py --isiolimited
echo "Running CPU limited benchmark"
python video_reading_benchmarks/main.py 

