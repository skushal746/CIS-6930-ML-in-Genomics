#!/bin/bash

# 1. Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv is not installed. Please install it first (e.g., curl -LsSf https://astral.sh/uv/install.sh | sh)."
    exit 1
fi

echo "--- Initializing uv environment ---"

# 2. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "--- Creating virtual environment (.venv) ---"
    uv venv
else
    echo "--- .venv already exists. Skipping creation. ---"
fi

# 3. Activate the environment
source .venv/bin/activate

# 4. Install requirements
if [ -f "requirements.txt" ]; then
    echo "--- Installing packages from requirements.txt ---"
    uv pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# 5. Link to Jupyter
echo "--- Linking environment to Jupyter ---"
python -m ipykernel install --user --name=ml-genomics --display-name "Python (Genomics)"

echo "--- Setup Complete! ---"
echo "To activate your environment manually, run: source .venv/bin/activate"
