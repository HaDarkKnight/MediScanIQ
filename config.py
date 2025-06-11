import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Secret key for session management and CSRF protection
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')

    # Directory for storing uploaded X-ray images
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')

    # Allowed file extensions for uploads
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    # Maximum upload size (16MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024