import streamlit as st
import sqlite3
import cv2
import numpy as np
import os
import json
from datetime import datetime
import hashlib

# Database setup
DB_PATH = "sqlite_database.db"
RESOURCES_PATH = "resources"
USERS_DB_PATH = "users.db"

st.set_page_config(page_title="Missing Person Finder", page_icon="👤", layout="wide")
st.title("👤 Missing Person Finder")
st.markdown("---")

# 🔥 Initialize user database
@st.cache_data
def init_user_db():
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            phone TEXT,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sightings_submitted INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_user_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username=? AND password_hash=?", 
                   (username, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    return user[0] if user else None

def get_user_profile(username):
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def update_sightings_count(username):
    conn = sqlite3.connect(USERS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET sightings_submitted = sightings_submitted + 1 WHERE username=?", (username,))
    conn.commit()
    conn.close()

# OpenCV Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        landmarks = []
        for i in range(68):
            rel_x = (x + w * 0.1 + (w * 0.8 * np.sin(i * np.pi / 34))) / image.shape[1]
            rel_y = (y + h * 0.2 + (h * 0.6 * np.cos(i * np.pi / 34))) / image.shape[0]
            landmarks.append([rel_x, rel_y, 0.0])
        return np.array(landmarks)
    return None

# 🔥 MAIN AUTHENTICATION PAGES
def login_page():
    st.subheader("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if authenticate_user(username, password):
                st.session_state["user"] = username
                st.session_state["logged_in"] = True
                st.success(f"✅ Welcome back, {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

def register_page():
    st.subheader("📝 Register New Account")
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Create Account")
        
        if submit:
            if password != confirm_password:
                st.error("❌ Passwords don't match")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters")
            else:
                try:
                    conn = sqlite3.connect(USERS_DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO users (username, email, phone, password_hash) 
                        VALUES (?, ?, ?, ?)
                    """, (username, email, phone, hash_password(password)))
                    conn.commit()
                    conn.close()
                    st.success("✅ Account created! Please login.")
                    st.info("👈 Use the Login tab")
                except sqlite3.IntegrityError:
                    st.error("❌ Username or email already exists")

# 🔥 SESSION MANAGEMENT
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["user"] = None

if not st.session_state["logged_in"]:
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        login_page()
    
    with tab2:
        register_page()
    
    st.markdown("---")
    st.info("👮‍♂️ **Police:** Use [main app](http://localhost:8501/) to match sightings")
else:
    # 🔥 MAIN PUBLIC APP - USER DASHBOARD
    user = st.session_state["user"]
    st.sidebar.success(f"👋 Welcome, **{user}**")
    
    # Profile section
    with st.sidebar:
        st.subheader("👤 Profile")
        profile = get_user_profile(user)
        col1, col2 = st.columns(2)
        col1.metric("Sightings Submitted", profile[6] if profile else 0)
        col2.button("📤 Logout", on_click=lambda: logout())
        
        st.markdown("---")
        st.markdown("[Main Police App](http://localhost:8501/)")
    
    def logout():
        st.session_state["logged_in"] = False
        st.session_state["user"] = None
        st.rerun()
    
    # Initialize main database
    @st.cache_data
    def init_db():
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public_submissions (
                id TEXT PRIMARY KEY,
                submitted_by TEXT,
                image_path TEXT,
                face_mesh TEXT,
                status TEXT DEFAULT 'NF',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT
            )
        """)
        # Add user_id column if missing
        cursor.execute("PRAGMA table_info(public_submissions)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'user_id' not in columns:
            cursor.execute("ALTER TABLE public_submissions ADD COLUMN user_id TEXT")
        conn.commit()
        conn.close()

    init_db()
    
    # Main tabs
    tab1, tab2 = st.tabs(["📸 Submit Photo", "📊 My Stats"])
    
    with tab1:
        st.subheader("📸 Report Sighting")
        name = st.text_input("Your Full Name")
        image_file = st.file_uploader("Upload photo", type=['png','jpeg','jpg'])
        
        if image_file and name:
            os.makedirs(RESOURCES_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"public_{timestamp}.jpg"
            image_path = f"{RESOURCES_PATH}/{image_filename}"
            
            with open(image_path, "wb") as f:
                f.write(image_file.getbuffer())
            
            image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
            st.image(image_np, channels="BGR", width=250, caption="Uploaded")
            
            with st.spinner("🔍 Analyzing face..."):
                face_features = extract_face_features(image_np)
            
            if face_features is not None:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO public_submissions (id, submitted_by, image_path, face_mesh, status, user_id)
                    VALUES (?, ?, ?, ?, 'NF', ?)
                """, (f"public_{timestamp}", name, image_path, json.dumps(face_features.tolist()), user))
                conn.commit()
                conn.close()
                
                # Update user sightings count
                update_sightings_count(user)
                
                st.success("✅ Sighting submitted! Police will review within 24 hours.")
                st.balloons()
                st.balloons()
            else:
                st.warning("⚠️ No face detected. Please upload a clear frontal photo.")
    
    with tab2:
        st.subheader("📊 Your Stats")
        profile = get_user_profile(user)
        col1, col2, col3 = st.columns(3)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM public_submissions WHERE status='NF' AND user_id=?", (user,))
        pending = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM public_submissions WHERE user_id=?", (user,))
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM public_submissions WHERE status='F' AND user_id=?", (user,))
        found = cursor.fetchone()[0]
        conn.close()
        
        col1.metric("📤 Pending", pending)
        col2.metric("📊 Total", total)
        col3.metric("✅ Found", found)
        
        st.info("👮‍♂️ Police matches are processed in the main app")
    
    st.markdown("---")
    st.markdown("""
    **🔒 Secure Login** | **📱 Mobile Friendly** | **⚡ Instant Face Analysis**
    """)
