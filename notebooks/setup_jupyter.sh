# Step 1: Create and Activate Virtual Environment (Recommended)
echo "Creating a virtual environment (jupyter_env)..."
python3 -m venv jupyter_env
source jupyter_env/bin/activate

# Step 2: Install Jupyter (If Not Installed)
echo "Installing Jupyter..."
pip install --upgrade pip
pip install jupyter ipykernel

# Step 3: Add Python 3.12 as a Jupyter Kernel
echo "Adding Python 3.12 as a Jupyter kernel..."
python -m ipykernel install --user --name=jupyter_env --display-name "Python 3.12 (jupyter_env)"


# Step 4: Launch Jupyter Notebook
echo "Launching Jupyter Notebook..."
jupyter notebook

