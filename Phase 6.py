import os
import jwt
import json
import logging
import hashlib
from flask import Flask, request, jsonify, make_response
from cryptography.fernet import Fernet
from google.cloud import kms
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.encryption import Algorithm


app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- SECURITY & ENCRYPTION ---

# Load KMS Client for Key Management
kms_client = kms.KeyManagementServiceClient()

# Generate a secure key for AES-256 Encryption
fernet_key = Fernet.generate_key()
fernet = Fernet(fernet_key)


def encrypt_data(data):
    """
    Encrypt sensitive data using AES-256 encryption.
    """
    return fernet.encrypt(data.encode())


def decrypt_data(token):
    """
    Decrypt sensitive data using AES-256 encryption.
    """
    return fernet.decrypt(token).decode()


# --- AUTHENTICATION ---

SECRET_KEY = "Your_JWT_Secret_Key"

def generate_jwt(user_id):
    """
    Generate a JWT Token for Authentication.
    """
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def verify_jwt(token):
    """
    Verify the provided JWT token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None


# --- DATABASE CONNECTION ---

client = MongoClient('mongodb://localhost:27017/')
db = client['fraud_detection']
logs = db['audit_logs']

def log_audit_event(event):
    """
    Store audit logs with immutability.
    """
    logs.insert_one({
        'event': event,
        'timestamp': datetime.utcnow()
    })


# --- GDPR COMPLIANCE ---

@app.route('/delete_user_data', methods=['POST'])
def delete_user_data():
    """
    Right to be Forgotten API.
    """
    data = request.json
    user_id = data.get('user_id')
    db.users.delete_one({'user_id': user_id})
    log_audit_event(f"User data deleted for user_id: {user_id}")
    return jsonify({'message': 'User data deleted successfully.'})


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    token = generate_jwt(user_id)
    log_audit_event(f"User {user_id} logged in.")

    return jsonify({'token': token})


@app.route('/secure_data', methods=['POST'])
def secure_data():
    data = request.json.get('data')
    encrypted_data = encrypt_data(data)
    log_audit_event(f"Sensitive data encrypted.")
    return jsonify({'encrypted_data': encrypted_data.decode()})


@app.route('/decrypt_data', methods=['POST'])
def decrypt_data_route():
    encrypted_data = request.json.get('encrypted_data')
    try:
        decrypted_data = decrypt_data(encrypted_data)
        log_audit_event(f"Sensitive data decrypted.")
        return jsonify({'decrypted_data': decrypted_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
