import os
from flask import Flask
from routes import register_routes
from db import init_db

# Initialize Flask app
app = Flask(__name__)

# Secret key for session management (replace with a secure key in production)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Ensure the database is initialized
init_db()

# Register all routes from routes.py
register_routes(app)

if __name__ == '__main__':
    # Debug mode for development; disable in production
    app.run(debug=True, host='0.0.0.0', port=5000)