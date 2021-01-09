#!/bin/bash
set -Eeuxo pipefail
for filename in assets/video_1920x1080.mkv assets/video_480x270.mkv # assets/video_720x480.mkv 
	do
		echo "Benchmark for $filename"
		echo "Running unblocked (no consumer bottleneck) benchmark"
		python video_reading_benchmarks/main.py --isiolimited --duration 0 --inputvideo $filename
		echo "Running IO limited benchmark"
		python video_reading_benchmarks/main.py --isiolimited --inputvideo $filename
		echo "Running CPU limited benchmark"
		python video_reading_benchmarks/main.py --inputvideo $filename 
	done
