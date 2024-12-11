#!/bin/bash

# Exit on any error
set -e

echo "Starting Cluster Setup..."

# Install Python dependencies
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

echo "Upgrading PyTorch to latest nightly build..."
pip install --pre torch==2.6.0.dev20241203+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade


# # Clone Apex repository... 
# # Apex's FusedRMSNorm is slower than pytorch's F.rms_norm

# echo "Cloning NVIDIA Apex repository..."
# if [ -d "apex" ]; then
#     echo "Removing existing apex directory..."
#     rm -rf apex
# fi
# git clone https://github.com/NVIDIA/apex
# cd apex

# # Install Apex
# echo "Installing Apex... This may take a few minutes.."
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# echo "Testing Apex installation..."
# python3 -c "from apex.normalization import FusedRMSNorm; print('Apex installation successful!')"

echo "Cluster Setup Complete!"