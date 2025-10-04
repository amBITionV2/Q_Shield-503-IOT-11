import os
import base64
import json
import logging
from datetime import datetime, timedelta

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import jwt

try:
    import oqs
except ImportError as e:
    raise ImportError(
        "oqs (liboqs-python) is required for Kyber512. "
        "Install it with:\n\n"
        "  git clone --depth=1 https://github.com/open-quantum-safe/liboqs-python.git\n"
        "  cd liboqs-python\n"
        "  pip install .\n"
    ) from e

# ♡ Config
JWT_SECRET = os.environ.get("JWT_SECRET", "qshield_jwt_secret")
JWT_ALG = "HS256"

# ♡ jWT helpers
def create_jwt(payload: dict, expires_minutes: int = 60) -> str:
    p = payload.copy()
    p["exp"] = datetime.utcnow() + timedelta(minutes=expires_minutes)
    return jwt.encode(p, JWT_SECRET, algorithm=JWT_ALG)

def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise Exception("JWT expired")
    except jwt.InvalidTokenError as e:
        raise Exception(f"Invalid JWT: {e}")

# ♡ AES-GCM helpers 
def encrypt_aes_gcm(key: bytes, plaintext: bytes) -> bytes:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("AES key must be bytes")
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext, None)
    return nonce + ct

def decrypt_aes_gcm(key: bytes, data: bytes) -> bytes:
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("AES key must be bytes")
    nonce, ct = data[:12], data[12:]
    aes = AESGCM(key)
    return aes.decrypt(nonce, ct, None)

# ♡ Kyber512 helpers 
def kyber_generate_keypair():
    with oqs.KeyEncapsulation("Kyber512") as kem:
        public_key = kem.generate_keypair()
        secret_key = kem.export_secret_key()
    return public_key, secret_key

def kyber_encapsulate(public_key: bytes):
    with oqs.KeyEncapsulation("Kyber512") as kem:
        ciphertext, shared_secret = kem.encap_secret(public_key)
    return ciphertext, shared_secret

def kyber_decapsulate(ciphertext: bytes, secret_key: bytes):
    with oqs.KeyEncapsulation("Kyber512", secret_key) as kem:
        shared_secret = kem.decap_secret(ciphertext)
    return shared_secret
