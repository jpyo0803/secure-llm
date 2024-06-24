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
execution_modes=("mode8" "mode9")
input_lengths=(128 256 512 1024)
batch_sizes=(1 12 24)

# Function to calculate the output lengths based on input length
get_output_lengths() {
  local input_length=$1
  local output_lengths=()
  local current_length=$((input_length * 2))

  while [ $current_length -le 2048 ]; do
    output_lengths+=($current_length)
    current_length=$((current_length * 2))
  done

  echo "${output_lengths[@]}"
}

# Iterate over all combinations of the arguments
for mode in "${execution_modes[@]}"; do
  for input_length in "${input_lengths[@]}"; do
    output_lengths=($(get_output_lengths $input_length))
    for output_length in "${output_lengths[@]}"; do
      for batch_size in "${batch_sizes[@]}"; do
        # Construct and run the command with sudo -E
        command="sudo -E python3 smoothquant_generation.py $mode $input_length $output_length $batch_size"
        echo "Running: $command"
        $command
      done
    done
  done
done

# Kill the keep_sudo_alive background process
kill $KEEP_SUDO_ALIVE_PID
