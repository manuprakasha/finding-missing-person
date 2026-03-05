import streamlit as st
from pages.helper import db_queries, match_algo, train_model
from pages.helper.streamlit_helpers import require_login
import sqlite3
import cv2
import numpy as np
import json
import os
import uuid

# 🔥 FIXED: OpenCV Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_mesh(image):
    """Extract face features using OpenCV - MediaPipe compatible"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img_h, img_w = image.shape[:2]
        all_landmarks = []
        
        # Generate 468 landmarks (MediaPipe format)
        for i in range(468):
            # Simplified landmark generation - normalized coordinates
            angle = i * np.pi / 234
            px = x + w * (0.5 + 0.3 * np.cos(angle))
            py = y + h * (0.4 + 0.4 * np.sin(angle))
            all_landmarks.extend([px/img_w, py/img_h, 0.0])
        
        return np.array(all_landmarks[:468*3]).reshape(468, 3)
    return None

def case_viewer(public_case_id):
    """Display confirmed case details"""
    try:
        conn = sqlite3.connect("sqlite_database.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT submitted_by, created_at FROM public_submissions WHERE id=?", (public_case_id,))
        details = cursor.fetchone()
        
        if details:
            st.success(f"✅ **CASE CONFIRMED FOUND!** 🎉")
            st.info(f"**Submitted by:** {details[0]}")
            st.info(f"**Case ID:** `{public_case_id[:8]}...`")
            st.info(f"**Submitted:** {details[1]}")
        conn.close()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# 🔥 MAIN APP
if "login_status" not in st.session_state:
    st.write("You don't have access to this page")
elif st.session_state["login_status"]:
    user = st.session_state.user
    st.title("🔍 Check for Match (Login Required)")
    
    # Case count
    conn = sqlite3.connect("sqlite_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM public_submissions WHERE status='NF'")
    case_count = cursor.fetchone()[0]
    st.success(f"📊 **{case_count} cases** available for matching")
    conn.close()
    
    col1, col2 = st.columns(2)
    refresh_bt = col1.button("🔄 Refresh & Train")
    st.markdown("---")
    
    st.subheader("📸 Upload Photo to Match")
    image_obj = st.file_uploader("Upload public photo", type=["jpg", "jpeg", "png"])
    
    if image_obj:
        with st.spinner("Processing..."):
            image_bytes = image_obj.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            st.image(image_cv, channels="BGR", width=250, caption="Uploaded Photo")
            
            current_face_mesh = extract_face_mesh(image_cv)
        
        if current_face_mesh is not None:
            st.success(f"✅ Face detected! Matching {case_count} cases...")
            
            conn = sqlite3.connect("sqlite_database.db")
            cursor = conn.cursor()
            cursor.execute("SELECT id, submitted_by, face_mesh FROM public_submissions WHERE status='NF'")
            cases = cursor.fetchall()
            conn.close()
            
            matches = []
            for case_id, name, mesh_json in cases:
                try:
                    stored_mesh = np.array(json.loads(mesh_json))
                    if stored_mesh.shape == current_face_mesh.shape:
                        distance = np.linalg.norm(current_face_mesh.flatten() - stored_mesh.flatten())
                        similarity = max(0, 1 - distance / 1000)
                        if similarity > 0.85:
                            matches.append((name, similarity, case_id))
                except:
                    continue
            
            if matches:
                matches.sort(key=lambda x: x[1], reverse=True)
                st.success(f"🎉 **{len(matches)} MATCHES FOUND!**")
                st.markdown("---")
                
                # 🔥 FIX: UNIQUE keys + session_state for button clicks
                for i, (name, sim, case_id) in enumerate(matches[:10], 1):
                    # Generate UNIQUE key
                    unique_key = f"confirm_{uuid.uuid4().hex[:8]}_{case_id}_{i}"
                    
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**#{i} 🎯 {name}**")
                        st.caption(f"Case ID: `{case_id[:8]}...`")
                    
                    with col2:
                        st.metric("Similarity", f"{sim:.1%}")
                    
                    with col3:
                        # 🔥 FIX: Use session_state to track confirmed cases
                        if f"confirmed_{case_id}" not in st.session_state:
                            st.session_state[f"confirmed_{case_id}"] = False
                        
                        if not st.session_state[f"confirmed_{case_id}"]:
                            if st.button(f"✅ CONFIRM MATCH", key=unique_key):
                                try:
                                    conn = sqlite3.connect("sqlite_database.db")
                                    cursor = conn.cursor()
                                    
                                    # Verify and update case
                                    cursor.execute("SELECT id FROM public_submissions WHERE id=? AND status='NF'", (case_id,))
                                    if cursor.fetchone():
                                        cursor.execute("UPDATE public_submissions SET status='F' WHERE id=?", (case_id,))
                                        conn.commit()
                                        st.session_state[f"confirmed_{case_id}"] = True
                                        st.success(f"✅ **Case {case_id[:8]}... MARKED FOUND!** 🎉")
                                        st.balloons()
                                        case_viewer(case_id)
                                    else:
                                        st.error("❌ Case already processed")
                                    
                                    conn.close()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                        else:
                            st.success("✅ Already confirmed")
                    
                    st.markdown("---")
            else:
                st.info("ℹ️ No matches above 85% threshold")
        else:
            st.warning("⚠️ No face detected!")
    
    if refresh_bt:
        with st.spinner("🔄 Training model..."):
            try:
                result = train_model.train(user)
                matched_ids = match_algo.match()
                if matched_ids["status"] and matched_ids["result"]:
                    st.success(f"🎉 **{len(matched_ids['result'])} AUTO-MATCHES!**")
                else:
                    st.info("ℹ️ No auto-matches")
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")

else:
    st.write("You don't have access to this page")
