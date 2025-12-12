#!/bin/bash

# Run interactive bash shell in Docker container
# Useful for debugging and running Python scripts directly

echo "════════════════════════════════════════════════════════════"
echo "Starting Interactive Bash Shell"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Container: gluonts-covid"
echo "Working directory: /workspace"
echo ""
echo "Type 'exit' to leave the container"
echo "════════════════════════════════════════════════════════════"
echo ""

docker run -it --rm \
    -v "$(pwd)":/workspace \
    --name gluonts-bash \
    gluonts-covid \
    /bin/bash

# Note:
# - Current directory is mounted to /workspace in container
# - Container is removed after exiting (--rm flag)
# - Interactive terminal with bash shell

