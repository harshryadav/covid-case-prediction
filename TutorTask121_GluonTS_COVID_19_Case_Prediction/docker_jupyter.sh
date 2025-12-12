#!/bin/bash

# Run Jupyter Notebook in Docker container
# Access at: http://localhost:8888

echo "════════════════════════════════════════════════════════════"
echo "Starting Jupyter Notebook Server"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Container: gluonts-covid"
echo "Port: 8888"
echo "URL: http://localhost:8888"
echo ""
echo "Press Ctrl+C to stop the server"
echo "════════════════════════════════════════════════════════════"
echo ""

docker run -it --rm \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    --name gluonts-jupyter \
    gluonts-covid

# Note: 
# - Port 8888 is mapped to localhost:8888
# - Current directory is mounted to /workspace in container
# - Container is removed after stopping (--rm flag)
# - No token/password required for local access

