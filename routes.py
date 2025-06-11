import os
import logging
import re
import random
import uuid
from datetime import datetime, timedelta
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
from flask import render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from db import init_db

# Disable oneDNN to suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONTACT_EMAIL = 'mediscaniq@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'mediscaniq@gmail.com'
SMTP_PASSWORD = 'ttuf pdhf yqya nqst'


# Directories for storing images
UPLOAD_DIR = os.path.join('static', 'uploads')
PROFILE_DIR = os.path.join('static', 'uploads', 'profiles')
for directory in [UPLOAD_DIR, PROFILE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

# Load the ML model
model_path = os.path.join('models', 'best_model.keras')
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
try:
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Enhanced class_info with detailed medicines and treatment for doctors
class_info = {
    'COVID19': {
        'description': 'COVID-19 is a highly contagious respiratory illness caused by the SARS-CoV-2 virus, first identified in December 2019 in Wuhan, China. It spreads primarily through respiratory droplets and can range from asymptomatic cases to severe illness. Common symptoms include high fever, persistent dry cough, shortness of breath, fatigue, and loss of taste or smell. In severe cases, it may lead to acute respiratory distress syndrome (ARDS), pneumonia, or multi-organ failure, particularly in older adults or those with underlying health conditions such as diabetes or heart disease.',
        'treatment': 'Treatment focuses on supportive care, including rest, hydration, and fever management with acetaminophen. For mild cases, monitor symptoms and isolate to prevent spread. Antiviral medications like Paxlovid (nirmatrelvir 300 mg with ritonavir 100 mg, twice daily for 5 days) or Remdesivir (200 mg IV on day 1, then 100 mg daily for 4 days) are recommended for high-risk or hospitalized patients. Severe cases require oxygen therapy (via nasal cannula or mask, titrated to maintain SpO2 >94%), mechanical ventilation if ARDS develops, and corticosteroids like dexamethasone (6 mg IV or orally daily for up to 10 days) under medical supervision. Fluid management and anticoagulation (e.g., heparin 5000 IU SC every 8-12 hours) may be needed for complications. Consult a pulmonologist for tailored therapy.',
        'medicines': 'Antiviral options: Paxlovid (nirmatrelvir 300 mg with ritonavir 100 mg, twice daily for 5 days, contraindicated with certain CYP3A4 inhibitors); Remdesivir (200 mg IV on day 1, then 100 mg daily for 4 days, monitor renal function). Symptom relief: acetaminophen (500-1000 mg every 4-6 hours, max 4000 mg/day) or ibuprofen (200-400 mg every 6-8 hours, max 3200 mg/day) for fever/pain. Cough: dextromethorphan (10-20 mg every 4 hours, max 120 mg/day) or guaifenesin (200-400 mg every 4 hours). Severe cases: dexamethasone (6 mg daily for 10 days, taper if prolonged), heparin (5000 IU SC every 8-12 hours for prophylaxis), or tocilizumab (8 mg/kg IV once, max 800 mg) for cytokine storm. Ensure electrolyte balance (e.g., potassium 20 mEq IV if hypokalemic) and monitor liver/kidney function.'
    },
    'NORMAL': {
        'description': 'A normal chest X-ray indicates healthy lung tissue with no visible signs of infection, inflammation, or structural abnormalities. The lungs appear clear, with no evidence of fluid buildup, masses, or scarring. This suggests the absence of acute or chronic respiratory diseases at the time of imaging.',
        'treatment': 'No specific treatment is required as the lungs are healthy. Recommend annual health check-ups, flu vaccination (0.5 mL IM annually), and pneumococcal vaccine (PCV13 or PPSV23, per CDC guidelines) for prevention. Encourage smoking cessation if applicable, and promote a balanced diet rich in antioxidants (e.g., vitamins A, C, E) and regular aerobic exercise (30 minutes, 5 days/week) to maintain lung function.',
        'medicines': 'No medications are necessary. Preventive supplements: Vitamin D (1000-2000 IU daily with food) or Vitamin C (500-1000 mg daily) for general immunity, zinc (8-11 mg daily) for respiratory health, but only under medical advice. Avoid over-the-counter drugs unless symptomatic (e.g., loratadine 10 mg daily for allergies if needed).'
    },
    'PNEUMONIA': {
        'description': 'Pneumonia is an acute infection that inflames the alveoli (air sacs) in one or both lungs, often caused by bacteria (e.g., Streptococcus pneumoniae), viruses (e.g., influenza), or fungi (e.g., Pneumocystis jirovecii). Symptoms include productive cough with phlegm, high fever, chills, chest pain worsened by breathing, and shortness of breath. It can be community-acquired or hospital-acquired, with severity ranging from mild to life-threatening, especially in immunocompromised individuals or the elderly.',
        'treatment': 'Treatment varies by cause: Bacterial pneumonia requires antibiotics (e.g., amoxicillin 1 g thrice daily for 5-7 days or levofloxacin 500 mg daily for 7-10 days if penicillin-allergic). Viral pneumonia may use oseltamivir (75 mg twice daily for 5 days) if influenza-related. Fungal pneumonia needs antifungals (e.g., fluconazole 400 mg daily for 2-6 weeks). Supportive care includes oxygen therapy (2-5 L/min via nasal cannula if SpO2 <92%), hydration (IV fluids at 1-2 mL/kg/hour), and chest physiotherapy. Hospitalize severe cases for IV antibiotics (e.g., ceftriaxone 2 g daily) or ventilatory support if respiratory failure occurs.',
        'medicines': 'Bacterial: amoxicillin (500 mg thrice daily for 5-7 days) or azithromycin (500 mg day 1, then 250 mg daily for 4 days); levofloxacin (500-750 mg daily for 5-10 days) if resistant. Viral: oseltamivir (75 mg twice daily for 5 days). Fungal: fluconazole (200-400 mg daily for 2-6 weeks) or amphotericin B (0.5-1 mg/kg IV daily, monitor renal function). Symptom relief: ibuprofen (200-400 mg every 6-8 hours, max 3200 mg/day), dextromethorphan (10-20 mg every 4 hours, max 120 mg/day), or guaifenesin (200-400 mg every 4 hours). Severe: ceftriaxone (1-2 g IV daily) or vancomycin (15-20 mg/kg IV every 8-12 hours) if MRSA suspected.'
    },
    'TURBERCULOSIS': {
        'description': 'Tuberculosis (TB) is a chronic bacterial infection caused by Mycobacterium tuberculosis, primarily affecting the lungs but capable of spreading to other organs (e.g., kidneys, spine). It is transmitted through airborne droplets from coughing or sneezing. Symptoms include a persistent cough (often with blood), significant weight loss, night sweats, fever, and fatigue. Latent TB may remain asymptomatic, while active TB requires immediate treatment to prevent complications like lung damage or meningitis.',
        'treatment': 'Standard treatment involves a 6-month regimen: intensive phase (2 months) with isoniazid (300 mg daily), rifampin (600 mg daily), ethambutol (15-20 mg/kg daily), and pyrazinamide (20-25 mg/kg daily), followed by a continuation phase (4 months) with isoniazid and rifampin. Use Directly Observed Therapy (DOT) to ensure adherence. Drug-resistant TB requires 9-24 months with second-line drugs (e.g., bedaquiline 400 mg daily for 2 weeks, then 200 mg thrice weekly). Monitor liver function (ALT/AST monthly) and provide nutritional support (high-protein diet, 1.2-1.5 g/kg/day). Surgical intervention may be needed for complications like cavities.',
        'medicines': 'First-line: isoniazid (300 mg daily with pyridoxine 25-50 mg to prevent neuropathy), rifampin (600 mg daily, avoid alcohol), ethambutol (15-20 mg/kg daily, monitor vision), pyrazinamide (20-25 mg/kg daily, max 2 g, check uric acid). Continuation: isoniazid and rifampin for 4 months. Drug-resistant: bedaquiline (400 mg daily for 2 weeks, then 200 mg thrice weekly for 24 weeks), levofloxacin (500-1000 mg daily), or linezolid (600 mg daily). Supportive: pyridoxine (25-50 mg daily), multivitamins, and high-calorie supplements (e.g., 500-1000 kcal/day) if malnourished.'
    }
}

# Helper functions for heatmap generation
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

def compute_gradcam(model, img_array, layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(heatmap, img_array, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_array = np.uint8(img_array)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    superimposed_img = heatmap * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# Database and validation helpers
def get_db_connection():
    conn = sqlite3.connect('medical_app.db', timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Must contain uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Must contain lowercase letter."
    if not re.search(r"\d", password):
        return False, "Must contain number."
    if not re.search(r"[@$!%*?&]", password):
        return False, "Must contain special character."
    return True, ""

def validate_phone(phone):
    phone_regex = r"^\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$"
    return bool(re.match(phone_regex, phone)), "Invalid phone number format (e.g., +1234567890 or 123-456-7890)."

def validate_birth_date(birth_date):
    try:
        birth = datetime.strptime(birth_date, '%Y-%m-%d')
        today = datetime.now()
        if birth > today or birth < datetime(1900, 1, 1):
            return False, "Birth date must be between 1900 and today."
        return True, ""
    except ValueError:
        return False, "Invalid birth date format."

def validate_email(email):
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(email_regex, email)), "Invalid email address."

def generate_otp():
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

def send_email(to_email, from_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

# Initialize database before creating default admin
init_db()

# Initialize default admin user
def init_default_admin():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE username = ?", ('superadmin',))
    if not cursor.fetchone():
        try:
            cursor.execute(
                """
                INSERT INTO Users (
                    fname, lname, username, email, password_hash, phone,
                    birth_date, profile_image, role, is_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    'Super',
                    'Admin',
                    'superadmin',
                    'abuyasoor5@gmail.com',
                    generate_password_hash('Admin123!'),
                    '+1234567890',
                    '1970-01-01',
                    '/static/uploads/profiles/default_profile.png',
                    'admin',
                    1
                )
            )
            conn.commit()
            logger.info("Default admin user 'superadmin' created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating default admin: {e}")
    conn.close()

init_default_admin()

# Routes
def register_routes(app):
    @app.route('/')
    def index():
        if 'username' in session:
            username = session['username']
            role = session.get('role')
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT fname, lname, profile_image FROM Users WHERE username = ?", (username,))
                user = cursor.fetchone()
                if not user:
                    logger.error(f"User not found in database for index: {username}")
                    session.clear()
                    return redirect(url_for('signin'))

                diagnoses = []
                if role == 'patient':
                    cursor.execute("SELECT * FROM Diagnoses WHERE patient_id = ? ORDER BY id DESC LIMIT 5", (username,))
                    diagnoses = cursor.fetchall()
                elif role == 'doctor':
                    cursor.execute("SELECT * FROM Diagnoses WHERE created_by = ? ORDER BY id DESC LIMIT 5", (username,))
                    diagnoses = cursor.fetchall()
                elif role == 'admin':
                    cursor.execute("SELECT * FROM Diagnoses ORDER BY id DESC LIMIT 5", ())
                    diagnoses = cursor.fetchall()

                # Format diagnoses for display
                formatted_diagnoses = []
                for diag in diagnoses:
                    diag_dict = dict(diag)
                    confidence = float(diag['confidence']) if diag['confidence'] is not None else 0.0
                    diag_dict['confidence_pct'] = f"{confidence * 100:.2f}%"
                    formatted_diagnoses.append(diag_dict)

                user_dict = {
                    'fname': user['fname'],
                    'lname': user['lname'],
                    'profile_image': user['profile_image'],
                    'role': role
                }
                return render_template('index.html', user=user_dict, diagnoses=formatted_diagnoses)
            except sqlite3.Error as e:
                logger.error(f"Database error during index fetch for {username}: {e}")
                flash('Failed to load recent activity due to a server error.', 'error')
                return render_template('index.html')
            finally:
                conn.close()
        return render_template('index.html')

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            fname = request.form.get('fname')
            lname = request.form.get('lname')
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            phone = request.form.get('phone')
            birth_date = request.form.get('birth_date')
            role = request.form.get('role')
            doctor_id = request.form.get('doctor_id') if role == 'doctor' else None
            profile_image = request.files.get('profile_image')

            required_fields = {
                'fname': fname, 'lname': lname, 'username': username,
                'email': email, 'password': password, 'phone': phone,
                'birth_date': birth_date, 'role': role
            }
            for field_name, field_value in required_fields.items():
                if not field_value:
                    logger.warning(f"Missing required field during signup: {field_name}")
                    flash(f'{field_name.capitalize()} is required.', 'error')
                    return redirect(url_for('signup'))

            if role not in ['patient', 'doctor']:
                logger.warning(f"Invalid role selected during signup: {role}")
                flash('Invalid role selected. Only patient or doctor roles are allowed.', 'error')
                return redirect(url_for('signup'))

            valid, msg = validate_password(password)
            if not valid:
                logger.warning(f"Password validation failed for {username}: {msg}")
                flash(msg, 'error')
                return redirect(url_for('signup'))

            valid, msg = validate_email(email)
            if not valid:
                logger.warning(f"Email validation failed for {username}: {msg}")
                flash(msg, 'error')
                return redirect(url_for('signup'))

            valid, msg = validate_phone(phone)
            if not valid:
                logger.warning(f"Phone validation failed for {username}: {msg}")
                flash(msg, 'error')
                return redirect(url_for('signup'))

            valid, msg = validate_birth_date(birth_date)
            if not valid:
                logger.warning(f"Birth date validation failed for {username}: {msg}")
                flash(msg, 'error')
                return redirect(url_for('signup'))

            if role == 'doctor':
                if not doctor_id:
                    logger.warning(f"Doctor ID missing for doctor role signup: {username}")
                    flash('Doctor ID is required for doctor role.', 'error')
                    return redirect(url_for('signup'))
                conn = get_db_connection()
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT doctor_id FROM DoctorIDs WHERE doctor_id = ?", (doctor_id,))
                    if not cursor.fetchone():
                        logger.warning(f"Invalid Doctor ID during signup for {username}: {doctor_id}")
                        flash('Invalid Doctor ID. Please provide a valid ID.', 'error')
                        return redirect(url_for('signup'))
                    session['doctor_id'] = doctor_id
                except sqlite3.Error as e:
                    logger.error(f"Database error during DoctorID validation for {username}: {e}")
                    flash('Failed to validate Doctor ID due to a server error.', 'error')
                    return redirect(url_for('signup'))
                finally:
                    conn.close()

            profile_image_path = '/static/uploads/profiles/default_profile.png'
            if profile_image and profile_image.filename:
                try:
                    img = Image.open(io.BytesIO(profile_image.read()))
                    filename = f"profile_{uuid.uuid4().hex}.png"
                    save_path = os.path.join(PROFILE_DIR, filename)
                    img.save(save_path, 'PNG')
                    profile_image_path = f"/static/uploads/profiles/{filename}"
                    logger.info(f"Profile image saved to {save_path} for {username}")
                except Exception as e:
                    logger.error(f"Failed to save profile image for {username}: {e}")
                    flash('Failed to upload profile image. Using default image.', 'warning')

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                otp = generate_otp()
                expiry = datetime.now() + timedelta(minutes=10)
                cursor.execute(
                    """
                    INSERT INTO Users (
                        fname, lname, username, email, password_hash, phone,
                        birth_date, profile_image, role, verification_token, token_expiry
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fname, lname, username, email, generate_password_hash(password),
                        phone, birth_date, profile_image_path, role, otp, expiry
                    )
                )
                conn.commit()
                email_body = f"Your OTP for account verification is: {otp}\nThis OTP is valid for 10 minutes."
                email_sent = send_email(
                    to_email=email,
                    from_email=SMTP_USERNAME,
                    subject="MediScanIQ Account Verification OTP",
                    body=email_body
                )
                if email_sent:
                    flash(f'An OTP has been sent to {email}. Please verify to complete signup.', 'success')
                    return redirect(url_for('verify_otp', username=username))
                else:
                    logger.warning(f"Failed to send OTP email to {email} for {username}")
                    flash('Failed to send OTP. Please try again.', 'error')
                    cursor.execute("DELETE FROM Users WHERE username = ?", (username,))
                    conn.commit()
                    return redirect(url_for('signup'))
            except sqlite3.IntegrityError:
                logger.warning(f"Username or email already exists during signup: {username}, {email}")
                flash('Username or email already exists.', 'error')
                return redirect(url_for('signup'))
            except sqlite3.Error as e:
                logger.error(f"Database error during signup for {username}: {e}")
                flash('Failed to sign up due to a server error.', 'error')
                return redirect(url_for('signup'))
            finally:
                conn.close()
        return render_template('signup.html')

    @app.route('/verify_otp', methods=['GET', 'POST'])
    def verify_otp():
        if request.method == 'POST':
            username = request.form.get('username')
            otp = request.form.get('otp')
            logger.debug(f"OTP verification attempt for {username}")

            if not username or not otp:
                logger.warning("Username or OTP missing in verification attempt")
                flash('Username and OTP are required.', 'error')
                return redirect(url_for('verify_otp', username=username))

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM Users WHERE username = ?", (username,))
                user = cursor.fetchone()
                if not user:
                    logger.warning(f"User not found during OTP verification: {username}")
                    flash('Username not found.', 'error')
                    return redirect(url_for('verify_otp', username=username))

                if not user['verification_token'] or not user['token_expiry']:
                    logger.warning(f"No valid OTP found for {username}")
                    flash('No valid OTP found. Please sign up again.', 'error')
                    return redirect(url_for('signup'))

                if datetime.now() > datetime.strptime(user['token_expiry'], '%Y-%m-%d %H:%M:%S.%f'):
                    logger.warning(f"OTP expired for {username}")
                    flash('OTP has expired. Please sign up again.', 'error')
                    cursor.execute("DELETE FROM Users WHERE username = ?", (username,))
                    conn.commit()
                    return redirect(url_for('signup'))

                if user['verification_token'] != otp:
                    logger.warning(f"Invalid OTP for {username}")
                    flash('Invalid OTP.', 'error')
                    return redirect(url_for('verify_otp', username=username))

                cursor.execute(
                    "UPDATE Users SET is_verified = 1, verification_token = NULL, token_expiry = NULL WHERE username = ?",
                    (username,)
                )
                conn.commit()
                logger.info(f"Account verified successfully for {username}")
                flash('Account verified successfully! Please sign in.', 'success')
                return redirect(url_for('signin'))
            except sqlite3.Error as e:
                logger.error(f"Database error during OTP verification for {username}: {e}")
                flash('Failed to verify OTP due to a server error.', 'error')
                return redirect(url_for('verify_otp', username=username))
            finally:
                conn.close()
        username = request.args.get('username', '')
        return render_template('verify_otp.html', username=username)

    @app.route('/signin', methods=['GET', 'POST'])
    def signin():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            logger.debug(f"Sign-in attempt for username: {username}")

            if not username or not password:
                logger.warning("Username or password missing in sign-in attempt")
                flash('Username and password are required.', 'error')
                return redirect(url_for('signin'))

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM Users WHERE username = ?", (username,))
                user = cursor.fetchone()
                logger.debug(f"User fetched from database: {user}")

                if not user:
                    logger.warning(f"User not found: {username}")
                    flash('Invalid username or password!', 'error')
                    return redirect(url_for('signin'))

                if not user['is_verified']:
                    logger.warning(f"User not verified: {username}")
                    flash('Please verify your account before signing in.', 'error')
                    return redirect(url_for('signin'))

                if check_password_hash(user['password_hash'], password):
                    logger.info(f"Password verified for user: {username}")
                    session.clear()
                    session['username'] = username
                    session['role'] = user['role']
                    session['fname'] = user['fname']
                    session['lname'] = user['lname']
                    session['profile_image'] = user['profile_image']
                    if user['role'] == 'doctor':
                        cursor.execute("SELECT doctor_id FROM DoctorIDs WHERE doctor_id = ?", (session.get('doctor_id', ''),))
                        result = cursor.fetchone()
                        session['doctor_id'] = result['doctor_id'] if result else None
                    else:
                        session['doctor_id'] = None
                    logger.debug(f"Session after sign-in: {session}")
                    flash('Sign-in successful!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    logger.warning(f"Invalid password for user: {username}")
                    flash('Invalid username or password!', 'error')
                    return redirect(url_for('signin'))
            except sqlite3.Error as e:
                logger.error(f"Database error during sign-in for {username}: {e}")
                flash('Failed to sign in due to a server error.', 'error')
                return redirect(url_for('signin'))
            except KeyError as e:
                logger.error(f"Missing key in user data for {username}: {e}")
                flash('User data is incomplete. Please contact support.', 'error')
                return redirect(url_for('signin'))
            finally:
                conn.close()
        return render_template('signin.html')

    @app.route('/signout')
    def signout():
        logger.info(f"User signed out: {session.get('username', 'Unknown')}")
        session.clear()
        flash('You have been signed out.', 'success')
        return redirect(url_for('index'))

    @app.route('/profile')
    def profile():
        if 'username' not in session:
            logger.warning("Unauthorized profile access attempt: No user session")
            flash('Please sign in to view your profile.', 'error')
            return redirect(url_for('signin'))

        username = session['username']
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if not user:
                logger.error(f"User not found in database for profile: {username}")
                flash('User not found. Please sign in again.', 'error')
                session.clear()
                return redirect(url_for('signin'))

            # Ensure session data is up-to-date
            session['fname'] = user['fname']
            session['lname'] = user['lname']
            session['profile_image'] = user['profile_image']
            session['role'] = user['role']

            doctor_id = None
            if user['role'] == 'doctor':
                cursor.execute("SELECT doctor_id FROM DoctorIDs WHERE doctor_id = ?", (session.get('doctor_id', ''),))
                result = cursor.fetchone()
                doctor_id = result['doctor_id'] if result else None

            user_dict = {
                'fname': user['fname'],
                'lname': user['lname'],
                'username': user['username'],
                'email': user['email'],
                'phone': user['phone'],
                'birth_date': user['birth_date'],
                'role': user['role'],
                'profile_image': user['profile_image'],
                'doctor_id': doctor_id
            }
            logger.debug(f"Profile loaded for {username}")
            return render_template('profile.html', user=user_dict)
        except sqlite3.Error as e:
            logger.error(f"Database error during profile fetch for {username}: {e}")
            flash('Failed to load profile due to a server error.', 'error')
            return redirect(url_for('index'))
        finally:
            conn.close()

    @app.route('/edit_profile', methods=['GET', 'POST'])
    def edit_profile():
        if 'username' not in session:
            logger.warning("Unauthorized edit profile access attempt: No user session")
            flash('Please sign in to edit your profile.', 'error')
            return redirect(url_for('signin'))

        username = session['username']
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if not user:
                logger.error(f"User not found in database for edit profile: {username}")
                flash('User not found. Please sign in again.', 'error')
                session.clear()
                return redirect(url_for('signin'))

            if request.method == 'POST':
                fname = request.form.get('fname')
                lname = request.form.get('lname')
                email = request.form.get('email')
                phone = request.form.get('phone')
                birth_date = request.form.get('birth_date')
                doctor_id = request.form.get('doctor_id') if user['role'] == 'doctor' else None
                profile_image = request.files.get('profile_image')

                required_fields = {
                    'fname': fname, 'lname': lname, 'email': email,
                    'phone': phone, 'birth_date': birth_date
                }
                for field_name, field_value in required_fields.items():
                    if not field_value:
                        logger.warning(f"Missing required field during edit profile for {username}: {field_name}")
                        flash(f'{field_name.capitalize()} is required.', 'error')
                        return redirect(url_for('edit_profile'))

                valid, msg = validate_email(email)
                if not valid:
                    logger.warning(f"Email validation failed during edit profile for {username}: {msg}")
                    flash(msg, 'error')
                    return redirect(url_for('edit_profile'))

                valid, msg = validate_phone(phone)
                if not valid:
                    logger.warning(f"Phone validation failed during edit profile for {username}: {msg}")
                    flash(msg, 'error')
                    return redirect(url_for('edit_profile'))

                valid, msg = validate_birth_date(birth_date)
                if not valid:
                    logger.warning(f"Birth date validation failed during edit profile for {username}: {msg}")
                    flash(msg, 'error')
                    return redirect(url_for('edit_profile'))

                if user['role'] == 'doctor':
                    if not doctor_id:
                        logger.warning(f"Doctor ID missing during edit profile for {username}")
                        flash('Doctor ID is required for doctor role.', 'error')
                        return redirect(url_for('edit_profile'))
                    cursor.execute("SELECT doctor_id FROM DoctorIDs WHERE doctor_id = ?", (doctor_id,))
                    if not cursor.fetchone():
                        logger.warning(f"Invalid Doctor ID during edit profile for {username}: {doctor_id}")
                        flash('Invalid Doctor ID. Please provide a valid ID.', 'error')
                        return redirect(url_for('edit_profile'))

                profile_image_path = user['profile_image']
                if profile_image and profile_image.filename:
                    try:
                        img = Image.open(io.BytesIO(profile_image.read()))
                        filename = f"profile_{uuid.uuid4().hex}.png"
                        save_path = os.path.join(PROFILE_DIR, filename)
                        img.save(save_path, 'PNG')
                        profile_image_path = f"/static/uploads/profiles/{filename}"
                        logger.info(f"Profile image updated to {save_path} for {username}")
                    except Exception as e:
                        logger.error(f"Failed to save profile image for {username}: {e}")
                        flash('Failed to upload profile image. Keeping current image.', 'warning')

                cursor.execute(
                    "SELECT username FROM Users WHERE email = ? AND username != ?",
                    (email, username)
                )
                if cursor.fetchone():
                    logger.warning(f"Email already in use during edit profile for {username}: {email}")
                    flash('Email is already in use by another account.', 'error')
                    return redirect(url_for('edit_profile'))

                try:
                    cursor.execute(
                        """
                        UPDATE Users
                        SET fname = ?, lname = ?, email = ?, phone = ?,
                            birth_date = ?, profile_image = ?
                        WHERE username = ?
                        """,
                        (fname, lname, email, phone, birth_date, profile_image_path, username)
                    )
                    conn.commit()
                    session['fname'] = fname
                    session['lname'] = lname
                    session['profile_image'] = profile_image_path
                    if user['role'] == 'doctor':
                        session['doctor_id'] = doctor_id
                    logger.info(f"Profile updated successfully for {username}")
                    flash('Profile updated successfully!', 'success')
                    return redirect(url_for('profile'))
                except sqlite3.IntegrityError:
                    logger.warning(f"Email conflict during edit profile for {username}: {email}")
                    flash('Email is already in use by another account.', 'error')
                    return redirect(url_for('edit_profile'))
                except sqlite3.Error as e:
                    logger.error(f"Database error during profile update for {username}: {e}")
                    flash('Failed to update profile due to a server error.', 'error')
                    return redirect(url_for('edit_profile'))

            doctor_id = None
            if user['role'] == 'doctor':
                cursor.execute("SELECT doctor_id FROM DoctorIDs WHERE doctor_id = ?", (session.get('doctor_id', ''),))
                result = cursor.fetchone()
                doctor_id = result['doctor_id'] if result else None

            user_dict = {
                'fname': user['fname'],
                'lname': user['lname'],
                'username': user['username'],
                'email': user['email'],
                'phone': user['phone'],
                'birth_date': user['birth_date'],
                'role': user['role'],
                'profile_image': user['profile_image'],
                'doctor_id': doctor_id
            }
            return render_template('edit_profile.html', user=user_dict)
        except sqlite3.Error as e:
            logger.error(f"Database error during edit profile fetch for {username}: {e}")
            flash('Failed to load edit profile page due to a server error.', 'error')
            return redirect(url_for('profile'))
        finally:
            conn.close()

# In routes.py, replace the entire dashboard route section with:

    @app.route('/dashboard')
    def dashboard():
        if 'username' not in session:
            logger.warning("Unauthorized dashboard access attempt: No user session")
            flash('Please sign in to access the dashboard.', 'error')
            return redirect(url_for('signin'))

        username = session['username']
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT fname, lname, profile_image, role FROM Users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if not user:
                logger.error(f"User not found in database for dashboard: {username}")
                flash('User not found. Please sign in again.', 'error')
                session.clear()
                return redirect(url_for('signin'))

            # Update session data
            role = session.get('role', user['role'])
            fname = user['fname']
            lname = user['lname']
            profile_image = user['profile_image']

            user_dict = {
                'fname': fname,
                'lname': lname,
                'profile_image': profile_image,
                'role': role
            }

            diagnoses = []
            total_diagnoses = 0
            recent_diagnoses = []
            editable = False

            if role == 'patient':
                # Fetch all diagnoses for stats
                cursor.execute("SELECT COUNT(*) as count FROM Diagnoses WHERE patient_id = ?", (username,))
                total_diagnoses = cursor.fetchone()['count']
                # Fetch recent diagnoses (up to 5)
                cursor.execute("SELECT * FROM Diagnoses WHERE patient_id = ? ORDER BY id DESC LIMIT 5", (username,))
                recent_diagnoses = cursor.fetchall()
                cursor.execute("SELECT * FROM Diagnoses WHERE patient_id = ? ORDER BY id DESC", (username,))
                diagnoses = cursor.fetchall()
                editable = False

                # Format recent diagnoses
                formatted_recent_diagnoses = []
                for diag in recent_diagnoses:
                    diag_dict = dict(diag)
                    confidence = float(diag['confidence']) if diag['confidence'] is not None else 0.0
                    diag_dict['confidence_pct'] = f"{confidence * 100:.2f}%"
                    formatted_recent_diagnoses.append(diag_dict)

                user_dict['total_diagnoses'] = total_diagnoses
                return render_template(
                    'dashboards/patient_dashboard.html',
                    user=user_dict,
                    diagnoses=diagnoses,
                    recent_diagnoses=formatted_recent_diagnoses,
                    editable=editable
                )
            elif role == 'doctor':
                cursor.execute("SELECT COUNT(*) as count FROM Diagnoses WHERE created_by = ?", (username,))
                total_diagnoses = cursor.fetchone()['count']
                cursor.execute("SELECT * FROM Diagnoses WHERE created_by = ? ORDER BY id DESC LIMIT 5", (username,))
                recent_diagnoses = cursor.fetchall()
                cursor.execute("SELECT * FROM Diagnoses WHERE created_by = ? ORDER BY id DESC", (username,))
                diagnoses = cursor.fetchall()
                editable = True
                formatted_recent_diagnoses = []
                for diag in recent_diagnoses:
                    diag_dict = dict(diag)
                    confidence = float(diag['confidence']) if diag['confidence'] is not None else 0.0
                    diag_dict['confidence_pct'] = f"{confidence * 100:.2f}%"
                    formatted_recent_diagnoses.append(diag_dict)

                user_dict['total_diagnoses'] = total_diagnoses
                return render_template(
                    'dashboards/doctor_dashboard.html',
                    user=user_dict,
                    diagnoses=diagnoses,
                    recent_diagnoses=formatted_recent_diagnoses,
                    editable=editable
                )
            elif role == 'admin':
                cursor.execute("SELECT COUNT(*) as count FROM Diagnoses", ())
                total_diagnoses = cursor.fetchone()['count']
                cursor.execute("SELECT * FROM Diagnoses ORDER BY id DESC LIMIT 5", ())
                recent_diagnoses = cursor.fetchall()
                cursor.execute("SELECT * FROM Diagnoses ORDER BY id DESC", ())
                diagnoses = cursor.fetchall()
                editable = False

                formatted_recent_diagnoses = []
                for diag in recent_diagnoses:
                    diag_dict = dict(diag)
                    confidence = float(diag['confidence']) if diag['confidence'] is not None else 0.0
                    diag_dict['confidence_pct'] = f"{confidence * 100:.2f}%"
                    formatted_recent_diagnoses.append(diag_dict)

                user_dict['total_diagnoses'] = total_diagnoses
                return render_template(
                    'dashboards/admin_dashboard.html',
                    user=user_dict,
                    diagnoses=diagnoses,
                    recent_diagnoses=formatted_recent_diagnoses,
                    editable=editable
                )
        except sqlite3.Error as e:
            logger.error(f"Database error during dashboard fetch for {username}: {e}")
            flash('Failed to load dashboard due to a server error.', 'error')
            return redirect(url_for('index'))
        finally:
            conn.close()

    # Admin-specific routes (properly indented at the same level as other routes)
    @app.route('/admin/results')
    def admin_results():
        if 'username' not in session or session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.*, u1.username as created_by, u2.username as patient_id 
                FROM Diagnoses d
                LEFT JOIN Users u1 ON d.created_by = u1.username
                LEFT JOIN Users u2 ON d.patient_id = u2.username
            """)
            results = [dict(row) for row in cursor.fetchall()]
            return jsonify(results)
        except sqlite3.Error as e:
            logger.error(f"Database error in admin_results: {e}")
            return jsonify({'error': 'Database error'}), 500
        finally:
            conn.close()

    @app.route('/admin/delete_result/<int:id>', methods=['DELETE'])
    def admin_delete_result(id):
        if 'username' not in session or session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Diagnoses WHERE id = ?", (id,))
            conn.commit()
            return jsonify({'success': True})
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Delete error in admin_delete_result: {e}")
            return jsonify({'error': 'Database error'}), 500
        finally:
            conn.close()

                
        @app.route('/admin/delete_result/<int:id>', methods=['DELETE'])
        def admin_delete_result(id):
            if 'username' not in session or session.get('role') != 'admin':
                return jsonify({'error': 'Unauthorized'}), 401
            
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM Diagnoses WHERE id = ?", (id,))
                conn.commit()
                return jsonify({'success': True})
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Delete error in admin_delete_result: {e}")
                return jsonify({'error': 'Database error'}), 500
            finally:
                conn.close()    
    @app.route('/delete_user/<username>', methods=['POST'])
    def delete_user(username):
        if 'username' not in session or session.get('role') != 'admin':
            flash('Unauthorized action', 'error')
            return redirect(url_for('index'))

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            # Delete associated diagnoses first
            cursor.execute("DELETE FROM Diagnoses WHERE patient_id = ? OR created_by = ?", 
                        (username, username))
            # Then delete user
            cursor.execute("DELETE FROM Users WHERE username = ?", (username,))
            conn.commit()
            flash(f'User {username} deleted successfully', 'success')
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error deleting user {username}: {e}")
            flash('Failed to delete user', 'error')
        finally:
            conn.close()
        
        return redirect(url_for('admin_users'))

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if 'username' not in session:
            logger.warning("Unauthorized upload attempt: No user session")
            flash('Please sign in to upload an X-ray.', 'error')
            return redirect(url_for('signin'))

        if request.method == 'POST':
            if 'xray' not in request.files:
                logger.warning("No file uploaded in upload request")
                flash('No file uploaded.', 'error')
                return redirect(url_for('upload'))

            file = request.files['xray']
            try:
                img = Image.open(io.BytesIO(file.read())).convert('L')
                original = np.array(img)
                filename = f"xray_{uuid.uuid4().hex}.png"
                save_path = os.path.join(UPLOAD_DIR, filename)
                img.save(save_path)
                xray_image_path = f"/static/uploads/{filename}"
                logger.info(f"X-ray image saved to {save_path} for {session['username']}")

                img_resized = img.resize((150, 150))
                arr = np.array(img_resized) / 255.0
                arr = np.expand_dims(arr, axis=(0, -1))
            except Exception as e:
                logger.error(f"Invalid image during upload for {session['username']}: {e}")
                flash(f'Invalid image: {e}', 'error')
                return redirect(url_for('upload'))

            try:
                preds = model.predict(arr)
                idx = np.argmax(preds[0])
                conf = float(np.max(preds[0]))
                label = CLASS_NAMES[idx]

                try:
                    layer_name = get_last_conv_layer(model)
                    heatmap = compute_gradcam(model, arr, layer_name, pred_index=idx)
                    superimposed_img = superimpose_heatmap(heatmap, original)
                    heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
                    heatmap_save_path = os.path.join(UPLOAD_DIR, heatmap_filename)
                    plt.imsave(heatmap_save_path, superimposed_img)
                    heatmap_path = f"/static/uploads/{heatmap_filename}"
                    logger.info(f"Heatmap saved to {heatmap_save_path} for {session['username']}")
                except Exception as e:
                    logger.error(f"Failed to generate heatmap for {session['username']}: {e}")
                    heatmap_path = '/static/default_heatmap.png'

                # Save the result to the database
                conn = get_db_connection()
                try:
                    cursor = conn.cursor()
                    patient_id = session['username']
                    created_by = session['username']
                    cursor.execute(
                        """
                        INSERT INTO Diagnoses (
                            patient_id, diagnosis, confidence, description,
                            treatment, medicines, heatmap, xray_image, created_by
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            patient_id,
                            label,
                            conf,
                            class_info[label]['description'],
                            class_info[label]['treatment'],
                            class_info[label]['medicines'],
                            heatmap_path,
                            xray_image_path,
                            created_by
                        )
                    )
                    # Get the ID of the newly inserted diagnosis
                    diagnosis_id = cursor.lastrowid
                    conn.commit()
                    logger.info(
                        f"Upload result saved for patient_id={patient_id}, "
                        f"diagnosis={label}, confidence={conf}, diagnosis_id={diagnosis_id}"
                    )
                    flash('X-ray uploaded and diagnosis saved successfully!', 'success')
                    # Redirect to result page for patients, dashboard for others
                    if session.get('role') == 'patient' or session.get('role') == 'doctor':
                        return redirect(url_for('result', id=diagnosis_id))
                    return redirect(url_for('dashboard'))
                except sqlite3.Error as e:
                    logger.error(f"Database error saving upload result for {session['username']}: {e}")
                    flash('Failed to save diagnosis due to a database error.', 'error')
                    return redirect(url_for('upload'))
                finally:
                    conn.close()
            except Exception as e:
                logger.error(f"Prediction failed for {session['username']}: {e}")
                flash(f'Prediction failed: {e}', 'error')
                return redirect(url_for('upload'))

        return render_template('upload.html')

    @app.route('/result/<int:id>')
    def result(id):
        if 'username' not in session:
            logger.warning("Unauthorized result access attempt: No user session")
            flash('Please sign in to view results.', 'error')
            return redirect(url_for('signin'))

        username = session['username']
        role = session.get('role')

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            # Allow access to patient who owns the diagnosis, doctor who created it, or admin
            cursor.execute(
                """
                SELECT * FROM Diagnoses
                WHERE id = ? AND (
                    patient_id = ? OR created_by = ? OR ? = 'admin'
                )
                """,
                (id, username, username, role)
            )
            diag = cursor.fetchone()
            if not diag:
                logger.warning(
                    f"Diagnosis not found or unauthorized access for "
                    f"user={username}, role={role}, diagnosis_id={id}"
                )
                flash('Diagnosis not found or you do not have permission to view it.', 'error')
                return redirect(url_for('dashboard'))

            # Convert to dict and format confidence
            diag_dict = dict(diag)
            confidence = float(diag['confidence']) if diag['confidence'] is not None else 0.0
            diag_dict['confidence_pct'] = f"{confidence * 100:.2f}%"
            diag_dict['heatmap'] = diag['heatmap'] or '/static/default_heatmap.png'
            diag_dict['xray_image'] = diag['xray_image'] or '/static/default_xray.png'

            logger.debug(f"Rendering result for diagnosis_id={id}, user={username}")
            return render_template('results.html', diag=diag_dict)
        except sqlite3.Error as e:
            logger.error(f"Database error fetching result for id={id}, user={username}: {e}")
            flash('Failed to load diagnosis due to a server error.', 'error')
            return redirect(url_for('dashboard'))
        finally:
            conn.close()

    @app.route('/edit_diagnosis/<int:id>', methods=['GET', 'POST'])
    def edit_diagnosis(id):
        if 'username' not in session:
            logger.warning("Unauthorized edit diagnosis attempt: No user session")
            flash('Please sign in to edit a diagnosis.', 'error')
            return redirect(url_for('signin'))

        username = session['username']
        role = session.get('role')
        if role not in ['doctor', 'patient']:
            logger.warning(f"Unauthorized edit diagnosis attempt by {username} with role {role}")
            flash('Only doctors or patients can edit diagnoses.', 'error')
            return redirect(url_for('dashboard'))

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM Diagnoses
                WHERE id = ? AND (patient_id = ? OR created_by = ?)
                """,
                (id, username, username)
            )
            diag = cursor.fetchone()
            if not diag:
                logger.warning(
                    f"Diagnosis not found or unauthorized edit attempt for "
                    f"user={username}, diagnosis_id={id}"
                )
                flash('Diagnosis not found or you do not have permission to edit it.', 'error')
                return redirect(url_for('dashboard'))

            if request.method == 'POST':
                diagnosis = request.form.get('diagnosis')
                confidence = request.form.get('confidence')
                description = request.form.get('description')
                treatment = request.form.get('treatment')
                medicines = request.form.get('medicines')

                required_fields = {
                    'diagnosis': diagnosis,
                    'confidence': confidence,
                    'description': description,
                    'treatment': treatment,
                    'medicines': medicines
                }
                for field_name, field_value in required_fields.items():
                    if not field_value:
                        logger.warning(f"Missing required field during edit diagnosis for {username}: {field_name}")
                        flash(f'{field_name.capitalize()} is required.', 'error')
                        return redirect(url_for('edit_diagnosis', id=id))

                try:
                    confidence = float(confidence)
                    if not 0 <= confidence <= 1:
                        logger.warning(f"Invalid confidence value during edit diagnosis for {username}: {confidence}")
                        flash('Confidence must be between 0 and 1.', 'error')
                        return redirect(url_for('edit_diagnosis', id=id))
                except ValueError:
                    logger.warning(f"Invalid confidence format during edit diagnosis for {username}: {confidence}")
                    flash('Confidence must be a valid number.', 'error')
                    return redirect(url_for('edit_diagnosis', id=id))

                try:
                    cursor.execute(
                        """
                        UPDATE Diagnoses
                        SET diagnosis = ?, confidence = ?, description = ?,
                            treatment = ?, medicines = ?
                        WHERE id = ?
                        """,
                        (diagnosis, confidence, description, treatment, medicines, id)
                    )
                    conn.commit()
                    logger.info(f"Diagnosis id={id} updated successfully by {username}")
                    flash('Diagnosis updated successfully!', 'success')
                    return redirect(url_for('result', id=id))
                except sqlite3.Error as e:
                    logger.error(f"Database error during diagnosis update for id={id}, user={username}: {e}")
                    flash('Failed to update diagnosis due to a server error.', 'error')
                    return redirect(url_for('edit_diagnosis', id=id))

            diag_dict = dict(diag)
            return render_template('edit_diagnosis.html', diag=diag_dict)
        except sqlite3.Error as e:
            logger.error(f"Database error during edit diagnosis fetch for id={id}, user={username}: {e}")
            flash('Failed to load edit diagnosis page due to a server error.', 'error')
            return redirect(url_for('dashboard'))
        finally:
            conn.close()

    @app.route('/delete_result/<int:id>', methods=['POST'])
    def delete_result(id):
        if 'username' not in session:
            logger.warning("Unauthorized delete diagnosis attempt: No user session")
            flash('Please sign in to delete a diagnosis.', 'error')
            return redirect(url_for('signin'))

        username = session['username']
        role = session.get('role')
        if role not in ['doctor', 'patient']:
            logger.warning(f"Unauthorized delete diagnosis attempt by {username} with role {role}")
            flash('Only doctors or patients can delete diagnoses.', 'error')
            return redirect(url_for('dashboard'))

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM Diagnoses
                WHERE id = ? AND (patient_id = ? OR created_by = ?)
                """,
                (id, username, username)
            )
            diag = cursor.fetchone()
            if not diag:
                logger.warning(
                    f"Diagnosis not found or unauthorized delete attempt for "
                    f"user={username}, diagnosis_id={id}"
                )
                flash('Diagnosis not found or you do not have permission to delete it.', 'error')
                return redirect(url_for('dashboard'))

            cursor.execute("DELETE FROM Diagnoses WHERE id = ?", (id,))
            conn.commit()
            logger.info(f"Diagnosis id={id} deleted successfully by {username}")
            flash('Diagnosis deleted successfully!', 'success')
            return redirect(url_for('dashboard'))
        except sqlite3.Error as e:
            logger.error(f"Database error during diagnosis deletion for id={id}, user={username}: {e}")
            flash('Failed to delete diagnosis due to a server error.', 'error')
            return redirect(url_for('dashboard'))
        finally:
            conn.close()

    @app.route('/contact', methods=['GET', 'POST'])
    def contact():
        if request.method == 'POST':
            name = request.form.get('name')
            email = request.form.get('email')
            message = request.form.get('message')

            required_fields = {'name': name, 'email': email, 'message': message}
            for field_name, field_value in required_fields.items():
                if not field_value:
                    logger.warning(f"Missing required field in contact form: {field_name}")
                    flash(f'{field_name.capitalize()} is required.', 'error')
                    return redirect(url_for('contact'))

            valid, msg = validate_email(email)
            if not valid:
                logger.warning(f"Invalid email in contact form: {email}")
                flash(msg, 'error')
                return redirect(url_for('contact'))

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO ContactMessages (name, email, message)
                    VALUES (?, ?, ?)
                    """,
                    (name, email, message)
                )
                conn.commit()
                logger.info(f"Contact message saved from {email}")

                # Send confirmation email
                email_body = (
                    f"Dear {name},\n\nThank you for contacting MediScanIQ. "
                    f"We have received your message:\n\n{message}\n\n"
                    f"We will respond to you at {email} as soon as possible.\n\n"
                    f"Best regards,\nMediScanIQ Team"
                )
                email_sent = send_email(
                    to_email=email,
                    from_email=SMTP_USERNAME,
                    subject="MediScanIQ Contact Confirmation",
                    body=email_body
                )
                if not email_sent:
                    logger.warning(f"Failed to send contact confirmation email to {email}")
                    flash('Message saved, but failed to send confirmation email.', 'warning')
                else:
                    flash('Your message has been sent successfully!', 'success')
                return redirect(url_for('index'))
            except sqlite3.Error as e:
                logger.error(f"Database error saving contact message from {email}: {e}")
                flash('Failed to send message due to a server error.', 'error')
                return redirect(url_for('contact'))
            finally:
                conn.close()
        return render_template('contact.html')

    @app.route('/forgot_password', methods=['GET', 'POST'])
    def forgot_password():
        if request.method == 'POST':
            email = request.form.get('email')
            if not email:
                logger.warning("Email missing in forgot password request")
                flash('Email is required.', 'error')
                return redirect(url_for('forgot_password'))

            valid, msg = validate_email(email)
            if not valid:
                logger.warning(f"Invalid email in forgot password request: {email}")
                flash(msg, 'error')
                return redirect(url_for('forgot_password'))

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM Users WHERE email = ?", (email,))
                user = cursor.fetchone()
                if not user:
                    logger.warning(f"User not found for forgot password: {email}")
                    flash('No account found with that email address.', 'error')
                    return redirect(url_for('forgot_password'))

                otp = generate_otp()
                expiry = datetime.now() + timedelta(minutes=10)
                cursor.execute(
                    """
                    UPDATE Users
                    SET reset_token = ?, reset_token_expiry = ?
                    WHERE email = ?
                    """,
                    (otp, expiry, email)
                )
                conn.commit()

                email_body = f"Your OTP for password reset is: {otp}\nThis OTP is valid for 10 minutes."
                email_sent = send_email(
                    to_email=email,
                    from_email=SMTP_USERNAME,
                    subject="MediScanIQ Password Reset OTP",
                    body=email_body
                )
                if email_sent:
                    flash(f'An OTP has been sent to {email}. Please verify to reset your password.', 'success')
                    return redirect(url_for('reset_verify', email=email))
                else:
                    logger.warning(f"Failed to send reset OTP email to {email}")
                    flash('Failed to send OTP. Please try again.', 'error')
                    return redirect(url_for('forgot_password'))
            except sqlite3.Error as e:
                logger.error(f"Database error during forgot password for {email}: {e}")
                flash('Failed to process request due to a server error.', 'error')
                return redirect(url_for('forgot_password'))
            finally:
                conn.close()
        return render_template('forgot_password.html')

    @app.route('/reset_request', methods=['GET', 'POST'])
    def reset_request():
        if request.method == 'POST':
            email = request.form.get('email')
            if not email:
                logger.warning("Email missing in reset request")
                flash('Email is required.', 'error')
                return redirect(url_for('reset_request'))

            valid, msg = validate_email(email)
            if not valid:
                logger.warning(f"Invalid email in reset request: {email}")
                flash(msg, 'error')
                return redirect(url_for('reset_request'))

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM Users WHERE email = ?", (email,))
                user = cursor.fetchone()
                if not user:
                    logger.warning(f"User not found for reset request: {email}")
                    flash('No account found with that email address.', 'error')
                    return redirect(url_for('reset_request'))

                otp = generate_otp()
                expiry = datetime.now() + timedelta(minutes=10)
                cursor.execute(
                    """
                    UPDATE Users
                    SET reset_token = ?, reset_token_expiry = ?
                    WHERE email = ?
                    """,
                    (otp, expiry, email)
                )
                conn.commit()

                email_body = f"Your OTP for password reset is: {otp}\nThis OTP is valid for 10 minutes."
                email_sent = send_email(
                    to_email=email,
                    from_email=SMTP_USERNAME,
                    subject="MediScanIQ Password Reset OTP",
                    body=email_body
                )
                if email_sent:
                    flash(f'An OTP has been sent to {email}. Please verify to reset your password.', 'success')
                    return redirect(url_for('reset_verify', email=email))
                else:
                    logger.warning(f"Failed to send reset OTP email to {email}")
                    flash('Failed to send OTP. Please try again.', 'error')
                    return redirect(url_for('reset_request'))
            except sqlite3.Error as e:
                logger.error(f"Database error during reset request for {email}: {e}")
                flash('Failed to process request due to a server error.', 'error')
                return redirect(url_for('reset_request'))
            finally:
                conn.close()
        return render_template('reset_request.html')

    @app.route('/reset_verify', methods=['GET', 'POST'])
    def reset_verify():
        if request.method == 'POST':
            email = request.form.get('email')
            otp = request.form.get('otp')
            logger.debug(f"Reset OTP verification attempt for {email}")

            if not email or not otp:
                logger.warning("Email or OTP missing in reset verification attempt")
                flash('Email and OTP are required.', 'error')
                return redirect(url_for('reset_verify', email=email))

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM Users WHERE email = ?", (email,))
                user = cursor.fetchone()
                if not user:
                    logger.warning(f"User not found during reset verification: {email}")
                    flash('Email not found.', 'error')
                    return redirect(url_for('reset_verify', email=email))

                if not user['reset_token'] or not user['reset_token_expiry']:
                    logger.warning(f"No valid reset OTP found for {email}")
                    flash('No valid OTP found. Please request a new one.', 'error')
                    return redirect(url_for('reset_request'))

                if datetime.now() > datetime.strptime(user['reset_token_expiry'], '%Y-%m-%d %H:%M:%S.%f'):
                    logger.warning(f"Reset OTP expired for {email}")
                    flash('OTP has expired. Please request a new one.', 'error')
                    return redirect(url_for('reset_request'))

                if user['reset_token'] != otp:
                    logger.warning(f"Invalid reset OTP for {email}")
                    flash('Invalid OTP.', 'error')
                    return redirect(url_for('reset_verify', email=email))

                session['reset_email'] = email
                logger.info(f"Reset OTP verified successfully for {email}")
                flash('OTP verified successfully! Please enter your new password.', 'success')
                return redirect(url_for('reset_password'))
            except sqlite3.Error as e:
                logger.error(f"Database error during reset OTP verification for {email}: {e}")
                flash('Failed to verify OTP due to a server error.', 'error')
                return redirect(url_for('reset_verify', email=email))
            finally:
                conn.close()
        email = request.args.get('email', '')
        return render_template('reset_verify.html', email=email)

    @app.route('/reset_password', methods=['GET', 'POST'])
    def reset_password():
        if 'reset_email' not in session:
            logger.warning("Unauthorized reset password attempt: No reset email in session")
            flash('Please verify your email first.', 'error')
            return redirect(url_for('reset_request'))

        if request.method == 'POST':
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            email = session['reset_email']

            if not password or not confirm_password:
                logger.warning("Password or confirm password missing in reset password attempt")
                flash('Password and confirmation are required.', 'error')
                return redirect(url_for('reset_password'))

            if password != confirm_password:
                logger.warning(f"Passwords do not match for {email}")
                flash('Passwords do not match.', 'error')
                return redirect(url_for('reset_password'))

            valid, msg = validate_password(password)
            if not valid:
                logger.warning(f"Password validation failed for {email}: {msg}")
                flash(msg, 'error')
                return redirect(url_for('reset_password'))

            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE Users
                    SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL
                    WHERE email = ?
                    """,
                    (generate_password_hash(password), email)
                )
                conn.commit()
                session.pop('reset_email', None)
                logger.info(f"Password reset successfully for {email}")
                flash('Password reset successfully! Please sign in with your new password.', 'success')
                return redirect(url_for('signin'))
            except sqlite3.Error as e:
                logger.error(f"Database error during password reset for {email}: {e}")
                flash('Failed to reset password due to a server error.', 'error')
                return redirect(url_for('reset_password'))
            finally:
                conn.close()
        return render_template('reset_password.html')

    @app.route('/reset_password_confirm')
    def reset_password_confirm():
        return render_template('reset_password_confirm.html')

    @app.route('/security')
    def security():
        return render_template('security.html')
    
    @app.route('/admin_users')
    def admin_users():
        if 'username' not in session or session.get('role') != 'admin':
            logger.warning("Unauthorized admin users access attempt")
            flash('You do not have permission to access this page.', 'error')
            return redirect(url_for('index'))

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Users ORDER BY username")
            users = cursor.fetchall()
            logger.debug(f"Admin users page loaded with {len(users)} users")
            return render_template('admin_users.html', users=users)
        except sqlite3.Error as e:
            logger.error(f"Database error during admin users fetch: {e}")
            flash('Failed to load users due to a server error.', 'error')
            return redirect(url_for('dashboard'))
        finally:
            conn.close()

    @app.route('/admin_user_detail/<username>')
    def admin_user_detail(username):
        if 'username' not in session or session.get('role') != 'admin':
            logger.warning("Unauthorized admin user detail access attempt")
            flash('You do not have permission to access this page.', 'error')
            return redirect(url_for('index'))

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if not user:
                logger.warning(f"User not found for admin user detail: {username}")
                flash('User not found.', 'error')
                return redirect(url_for('admin_users'))

            cursor.execute("SELECT * FROM Diagnoses WHERE patient_id = ? OR created_by = ? ORDER BY id DESC", (username, username))
            diagnoses = cursor.fetchall()

            user_dict = dict(user)
            logger.debug(f"Admin user detail loaded for {username}")
            return render_template('admin_user_detail.html', user=user_dict, diagnoses=diagnoses)
        except sqlite3.Error as e:
            logger.error(f"Database error during admin user detail fetch for {username}: {e}")
            flash('Failed to load user details due to a server error.', 'error')
            return redirect(url_for('admin_users'))
        finally:
            conn.close()
        