#!/bin/bash

# Request sudo privileges upfront
sudo -v

# Function to refresh sudo timestamp periodically
keep_sudo_alive() {
  while true; do
    sudo -v
    sleep 60
  done
}

# Run the keep_sudo_alive function in the background
keep_sudo_alive &

# Get the PID of the background process
KEEP_SUDO_ALIVE_PID=$!

# Define the arrays of possible values for each argument
execution_modes=("mode6")
input_lengths=(1024 512 256 128)
batch_sizes=(24 12 1)

# Path to tc_malloc library
TCMALLOC_LIB="/usr/lib/x86_64-linux-gnu/libtcmalloc.so"

# Verify the library path
if [ ! -f "$TCMALLOC_LIB" ]; then
  echo "Error: $TCMALLOC_LIB not found."
  kill $KEEP_SUDO_ALIVE_PID
  exit 1
fi

# Export LD_PRELOAD for the entire script
export LD_PRELOAD=$TCMALLOC_LIB
echo "LD_PRELOAD set to $LD_PRELOAD"

# Iterate over all combinations of the arguments
for mode in "${execution_modes[@]}"; do
  for input_length in "${input_lengths[@]}"; do
    output_length=$((input_length * 2))
    for batch_size in "${batch_sizes[@]}"; do
      # Construct and run the command with sudo -E and LD_PRELOAD set for tc_malloc
      command="sudo -E LD_PRELOAD=$TCMALLOC_LIB python3 smoothquant_generation.py $mode $input_length $output_length $batch_size"
      echo "Running: $command"
      eval $command
    done
  done
done

# Kill the keep_sudo_alive background process
kill $KEEP_SUDO_ALIVE_PID
