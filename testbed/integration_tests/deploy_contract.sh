#!/bin/bash

# Read the input file line by line
while read -r line; do
  # Check if the line contains "Contract address:"
  if [[ $line =~ "Contract address:" ]]; then
    # Extract the string starting with "0x" in the same line
    contract_address=$(echo "$line" | grep -oP "0x\w+")
    # Print the contract address
    echo "$contract_address"
  fi
done < input_file.txt