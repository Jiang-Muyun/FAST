#!/bin/bash

cleanup() {
    echo "Caught SIGINT. Terminating all processes..."
    kill $PID1 $PID2 2>/dev/null
    wait $PID1 $PID2 2>/dev/null
    echo "All processes terminated."
    exit 1
}
trap cleanup SIGINT # Trap SIGINT (Ctrl-C)

# Start the processes in the background
python3 train_FAST_BCIC2020Track3.py --gpu 0 --folds "0-7"   &
PID1=$!
python3 train_FAST_BCIC2020Track3.py --gpu 1 --folds "7-15"  &
PID3=$!

wait $PID1 $PID2 # Wait for all processes to complete
