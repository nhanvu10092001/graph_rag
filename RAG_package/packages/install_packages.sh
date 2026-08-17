#!/bin/bash

# List of directories
directories=("llm-utils-core" "llm-utils-llm" "llm-utils-mcp" "llm-utils-vector" "llm-utils-rag" "llm-utils-subagent" "llm-utils-graph-rag" "llm-utils-parser" "llm-utils-reranker")

# Loop through each directory
for dir in "${directories[@]}"; do
  # Check if the directory exists
  if [ -d "$dir" ]; then
    # Change to the directory
    cd "$dir" || exit
    # Run pip install -e .
    echo "Installing in $dir"
    pip install -e .
    # Return to the base directory
    cd ..
  else
    echo "Directory $dir not found"
  fi
done
