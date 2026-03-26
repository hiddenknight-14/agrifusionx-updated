# setup.sh - Create this file in the project root
#!/bin/bash

echo "Setting up AgriFusionX - Leaf Disease Detection System"
echo "======================================================"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Run the application
python manage.py runserver

echo "======================================================"
echo "Setup complete! To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the server: python manage.py runserver"
echo "3. Open browser and go to: http://127.0.0.1:8000"
echo "======================================================"