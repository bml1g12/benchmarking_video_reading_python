#!/bin/bash

echo "Running unblocked (no consumer bottleneck) benchmark"
python video_reading_benchmarks/main.py --isiolimited --duration 0
echo "Running IO limited benchmark"
python video_reading_benchmarks/main.py --isiolimited
echo "Running CPU limited benchmark"
python video_reading_benchmarks/main.py 

