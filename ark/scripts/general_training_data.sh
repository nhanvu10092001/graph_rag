#!/bin/bash

# Script to combine training and validation data from amazon, prime, and mag into an 'all' folder

# Base directory
BASE_DIR="../data/experiments"
GENERAL_DIR="${BASE_DIR}/general/graph_explorer_gpt-4.1"

# Create the experiments/all directory structure
echo "Creating directory structure..."
mkdir -p "${GENERAL_DIR}/train"
mkdir -p "${GENERAL_DIR}/val"

# Function to copy files with prefix
copy_with_prefix() {
    local source_dir=$1
    local dest_dir=$2
    local prefix=$3
    
    if [ ! -d "$source_dir" ]; then
        echo "Warning: Source directory $source_dir does not exist, skipping..."
        return
    fi
    
    for file in "${source_dir}"/*; do
        if [ -e "$file" ]; then
            filename=$(basename "$file")
            cp -r "$file" "${dest_dir}/${prefix}_${filename}"
        fi
    done
}

# Copy train data from amazon, prime, and mag
echo "Copying train data from amazon..."
copy_with_prefix "${BASE_DIR}/amazon/graph_explorer_gpt-4.1/train" "${GENERAL_DIR}/train" "amazon"

echo "Copying train data from prime..."
copy_with_prefix "${BASE_DIR}/prime/graph_explorer_gpt-4.1/train" "${GENERAL_DIR}/train" "prime"

echo "Copying train data from mag..."
copy_with_prefix "${BASE_DIR}/mag/graph_explorer_gpt-4.1/train" "${GENERAL_DIR}/train" "mag"

# Copy val data from amazon, prime, and mag
echo "Copying val data from amazon..."
copy_with_prefix "${BASE_DIR}/amazon/graph_explorer_gpt-4.1/val" "${GENERAL_DIR}/val" "amazon"

echo "Copying val data from prime..."
copy_with_prefix "${BASE_DIR}/prime/graph_explorer_gpt-4.1/val" "${GENERAL_DIR}/val" "prime"

echo "Copying val data from mag..."
copy_with_prefix "${BASE_DIR}/mag/graph_explorer_gpt-4.1/val" "${GENERAL_DIR}/val" "mag"

echo "Done! Combined data is in ${GENERAL_DIR}/train and ${GENERAL_DIR}/val"

