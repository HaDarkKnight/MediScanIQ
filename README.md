# MediScanIQ: AI-Powered Lung Disease Classification System

## Overview
MediScanIQ is an AI-driven solution designed to assist healthcare professionals in diagnosing lung diseases from chest X-ray images. Leveraging a custom Convolutional Neural Network (CNN), it classifies images into four categories: COVID-19, Normal, Pneumonia, and Tuberculosis, achieving 93.54% accuracy. Integrated with a Flask-based web application, it offers secure, role-based access for patients, doctors, and administrators, enhancing diagnostic efficiency and accessibility.

## Features
- **AI Model**: Custom CNN trained on over 30,000 X-ray images for high accuracy.
- **Web Application**: Flask-based interface with OTP verification, X-ray upload, and Grad-CAM heatmaps for interpretability.
- **Roles**: Patient (upload/view diagnoses), Doctor (manage diagnoses), Admin (user/diagnosis management).
- **Technologies**: Python, TensorFlow/Keras, Flask, SQLite, SMTP.

## Project Structure
- `code/`: Python scripts (e.g., model training, web app routes).
- `models/`: Saved model file (e.g., `best_model.keras`).
- `docs/`: Documentation and reports.
- `data/`: Dataset info (note: actual images not included due to size; refer to Kaggle sources).

## Installation
1. Clone the repository: `git clone https://github.com/HaDarkKnight/MediScanIQ.git`
2. Navigate to the folder: `cd MediScanIQ`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the web app: `python app.py`

## Usage
- Upload a chest X-ray (PNG/JPG) via the web interface.
- View diagnosis results with confidence scores and Grad-CAM heatmaps.
- Access role-specific dashboards based on user login.

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. For major changes, open an issue to discuss first.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset sources: Kaggle, GitHub, Mendeley Data.
- Inspired by advancements in AI healthcare applications.
- Special thanks to team members MohammedAljammal and Abdullah Abu Fodeh, and supervisor KhalidAlemerien.

## Future Work
- Enhance model accuracy with larger datasets.
- Develop a mobile app for broader accessibility.
- Integrate with hospital systems for real-time use.
