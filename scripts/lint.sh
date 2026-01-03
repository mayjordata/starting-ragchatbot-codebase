#!/bin/bash
# Run code quality checks

set -e

echo "Running code quality checks..."
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Run ruff linter
echo "=== Running Ruff Linter ==="
uv run --extra dev ruff check .
echo "Ruff: OK"
echo ""

# Check black formatting
echo "=== Checking Black Formatting ==="
uv run --extra dev black --check .
echo "Black: OK"
echo ""

echo "All quality checks passed!"
