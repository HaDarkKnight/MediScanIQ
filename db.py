import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('medical_app.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fname TEXT NOT NULL,
            lname TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            phone TEXT NOT NULL,
            birth_date DATE NOT NULL,
            profile_image TEXT,
            role TEXT NOT NULL CHECK (role IN ('patient', 'doctor', 'admin')),
            is_verified BOOLEAN DEFAULT FALSE,
            verification_token TEXT,
            token_expiry DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Diagnoses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            confidence REAL NOT NULL,
            description TEXT,
            treatment TEXT,
            medicines TEXT,
            heatmap TEXT,
            xray_image TEXT,
            created_by TEXT NOT NULL,
            FOREIGN KEY (created_by) REFERENCES Users(username) ON DELETE RESTRICT ON UPDATE CASCADE        )
    """)

    # ContactMessages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ContactMessages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Activities table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES Users(user_id) ON DELETE CASCADE
        );
    """)

    # DoctorIDs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS DoctorIDs (
            doctor_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Insert default doctor IDs
    default_doctor_ids = [
        ('DOC001',),
        ('DOC002',),
        ('DOC003',),
        ('DOC004',),
        ('DOC005',)
    ]
    cursor.executemany("INSERT OR IGNORE INTO DoctorIDs (doctor_id) VALUES (?)", default_doctor_ids)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == '__main__':
    init_db()