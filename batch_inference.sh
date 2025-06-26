#!/bin/bash

# ======================================================================================
# This script runs batch face swapping inference based on a user-provided CSV file.
# It calls the 'batch_inference_image.py' script.
#
# The structure is based on the original 'inference_selected.sh' script by the author.
# ======================================================================================

# --- 1. User Configuration ---

# IMPORTANT: Please specify the path to your input CSV file.
# The CSV file must contain 'source' and 'target' columns
# pointing to the respective image paths.
CSV_PATH="image_pairs.csv"

# Specify the main directory for saving the output images and the result CSV.
Results_dir="batch_results"

# --- 2. Model and Device Configuration ---

# Set the GPU device to use
device=0

# Path to the model's configuration file.
CONFIG="models/REFace/configs/project_ffhq.yaml"

# Path to the model's checkpoint file.
# You can also use "models/REFace/checkpoints/last.ckpt" if needed.
CKPT="/c/huggingface_model/REFace/last.ckpt"

# --- 3. Inference Parameters ---

# Unconditional guidance scale.
scale=3.5

# Number of DDIM sampling steps.
ddim_steps=50

# --- 4. Pre-run Check ---
# Check if the CSV_PATH is set and the file exists.
if [ ! -f "$CSV_PATH" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! ERROR: CSV file not found at '$CSV_PATH'."
    echo "!!! Please update the CSV_PATH variable in this script."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    # Create a sample CSV file to guide the user.
    if [ ! -e "$CSV_PATH" ]; then
      echo "source,target" > "$CSV_PATH"
      echo "examples/faceswap/source.jpg,examples/faceswap/target.jpg" >> "$CSV_PATH"
      echo "A sample '$CSV_PATH' has been created. Please edit it with your file paths."
    fi
    exit 1
fi

# --- 5. Execute the Batch Inference Script ---
echo "Starting batch inference..."
echo "  - Input CSV: ${CSV_PATH}"
echo "  - Output Directory: ${Results_dir}"
echo "  - Config File: ${CONFIG}"
echo "  - Checkpoint File: ${CKPT}"
echo "  - Using Device: GPU ${device}"

CUDA_VISIBLE_DEVICES=${device} python scripts/batch_inference_image.py \
    --csv_path "${CSV_PATH}" \
    --outdir "${Results_dir}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --scale ${scale} \
    --ddim_steps ${ddim_steps}

echo "================================================="
echo "Batch inference finished successfully."
echo "Results are saved in the '${Results_dir}' directory."
echo "=================================================" 
