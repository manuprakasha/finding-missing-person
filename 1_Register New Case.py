import uuid
import numpy as np
import streamlit as st
import json
import base64
import os

from pages.helper.data_models import RegisteredCases
from pages.helper import db_queries
from pages.helper.utils import image_obj_to_numpy, extract_face_mesh_landmarks
from pages.helper.streamlit_helpers import require_login

st.set_page_config(page_title="Case New Form")

def image_to_base64(image):
    return base64.b64encode(image).decode("utf-8")

if "login_status" not in st.session_state:
    st.write("You don't have access to this page")
elif st.session_state["login_status"]:
    user = st.session_state.user
    st.title("Register New Case")
    
    # ✅ FIXED: Create these OUTSIDE columns so they're shared
    image_obj = None
    unique_id = None
    face_mesh = None
    save_flag = 0
    
    # Create resources folder if it doesn't exist
    os.makedirs("./resources", exist_ok=True)
    
    image_col, form_col = st.columns(2)
    
    with image_col:
        image_obj = st.file_uploader(
            "Image", type=["jpg", "jpeg", "png"], key="new_case"
        )
        
        if image_obj:
            # ✅ FIXED: Reset file pointer + generate ID
            image_obj.seek(0)
            unique_id = str(uuid.uuid4())
            uploaded_file_path = f"./resources/{unique_id}.jpg"
            
            # Save file properly
            with open(uploaded_file_path, "wb") as f:
                f.write(image_obj.getvalue())
            
            # Reset for display
            image_obj.seek(0)
            st.image(image_obj)
            
            # Process button - creates face_mesh
            if st.button("🔍 Process Face Mesh"):
                with st.spinner("Extracting face landmarks..."):
                    image_obj.seek(0)
                    image_numpy = image_obj_to_numpy(image_obj)
                    face_mesh = extract_face_mesh_landmarks(image_numpy)
                    
                    if face_mesh:
                        st.success("✅ Face mesh extracted successfully!")
                        st.session_state.face_mesh = face_mesh  # Store in session
                        st.session_state.unique_id = unique_id    # Store ID too
                    else:
                        st.error("❌ No face detected. Try another image.")
                        face_mesh = None
    
    # ✅ FIXED: Form uses session_state - works every time!
    with form_col:
        if image_obj:
            with st.form(key="new_case_form"):
                name = st.text_input("Name")
                fathers_name = st.text_input("Father's Name")
                age = st.number_input("Age", min_value=3, max_value=100, value=10, step=1)
                mobile_number = st.text_input("Mobile Number")
                address = st.text_input("Address")
                adhaar_card = st.text_input("Adhaar Card")
                birthmarks = st.text_input("Birth Mark")
                last_seen = st.text_input("Last Seen")
                description = st.text_area("Description (optional)")
                
                complainant_name = st.text_input("Complainant Name")
                complainant_phone = st.text_input("Complainant Phone")
                
                submit_bt = st.form_submit_button("💾 Save Case")
                
                if submit_bt:
                    # ✅ FIXED: Use session_state values
                    case_id = st.session_state.get('unique_id')
                    case_face_mesh = st.session_state.get('face_mesh')
                    
                    if case_id and case_face_mesh:
                        new_case_details = RegisteredCases(
                            id=case_id,
                            submitted_by=user,
                            name=name,
                            fathers_name=fathers_name,
                            age=age,
                            complainant_mobile=mobile_number,
                            complainant_name=complainant_name,
                            face_mesh=json.dumps(case_face_mesh),
                            adhaar_card=adhaar_card,
                            birth_marks=birthmarks,
                            address=address,
                            last_seen=last_seen,
                            status="NF",
                            matched_with="",
                        )
                        db_queries.register_new_case(new_case_details)
                        st.success("🎉 Case Registered Successfully!")
                        st.session_state.pop('face_mesh', None)  # Clear after save
                        st.session_state.pop('unique_id', None)
                    else:
                        st.error("❌ Please click 'Process Face Mesh' first!")
        else:
            st.info("👆 Please upload an image first")
else:
    st.write("You don't have access to this page")
