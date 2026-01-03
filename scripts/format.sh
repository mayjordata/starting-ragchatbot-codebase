#!/bin/bash
# Format code with black and fix ruff issues

set -e

echo "Formatting code..."
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Fix ruff issues (including import sorting)
echo "=== Fixing Ruff Issues ==="
uv run --extra dev ruff check --fix .
echo ""

# Format with black
echo "=== Formatting with Black ==="
uv run --extra dev black .
echo ""

echo "Formatting complete!"
