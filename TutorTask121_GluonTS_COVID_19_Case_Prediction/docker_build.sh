#!/bin/bash

# Build Docker image for GluonTS COVID-19 Project
# This script builds the Docker container with all dependencies

echo "════════════════════════════════════════════════════════════"
echo "Building Docker Image: gluonts-covid"
echo "════════════════════════════════════════════════════════════"
echo ""

docker build -t gluonts-covid .

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "✓ Docker image built successfully!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "  1. Run Jupyter: ./docker_jupyter.sh"
    echo "  2. Or run bash: ./docker_bash.sh"
    echo ""
else
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "✗ Docker build failed!"
    echo "════════════════════════════════════════════════════════════"
    exit 1
fi

