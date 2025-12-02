#!/bin/bash

# Script to run face recognition system with lower priority on STM32MP257

# Set CPU affinity to use only one core (if possible)
# This avoids interfering with other critical processes
if command -v taskset &> /dev/null; then
    echo "Setting CPU affinity to core 1"
    exec taskset -c 1 nice -n 10 python3 face_recognition_gui.py
else
    # If taskset not available, just use nice
    echo "Running with reduced priority"
    exec nice -n 10 python3 face_recognition_gui.py
fi