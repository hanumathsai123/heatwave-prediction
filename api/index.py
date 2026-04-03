import sys
import os
from pathlib import Path

# Add the heatwave_prediction directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'heatwave_prediction'))

# Import and initialize Flask app
from app import app as flask_app

# Export as handler for Vercel
app = flask_app
