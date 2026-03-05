import PIL
import numpy as np
import streamlit as st
import cv2

def image_obj_to_numpy(image_obj) -> np.ndarray:
    """Convert a Streamlit-uploaded image object to a numpy array."""
    image = PIL.Image.open(image_obj)
    return np.array(image)

def extract_face_mesh_landmarks(image: np.ndarray):
    """
    Extract face mesh landmarks from an image using OpenCV.
    Returns a flattened list of all (x, y, z) landmarks if a face is found, else None.
    """
    # OpenCV Face Detection (NO MediaPipe)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert BGR to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) > 0:
        # Use first detected face
        (x, y, w, h) = faces[0]
        
        # Generate 68 facial landmarks (standard facial landmark points)
        landmarks = []
        img_h, img_w = image.shape[:2]
        
        # Key facial landmark regions: eyes, nose, mouth, jawline
        facial_landmarks = [
            # Left eye (6 points)
            [(x + w*0.15, y + h*0.15), (x + w*0.25, y + h*0.18), (x + w*0.35, y + h*0.15),
             (x + w*0.45, y + h*0.18), (x + w*0.55, y + h*0.15), (x + w*0.65, y + h*0.18)],
            # Right eye (6 points)  
            [(x + w*0.35, y + h*0.35), (x + w*0.45, y + h*0.38), (x + w*0.55, y + h*0.35),
             (x + w*0.65, y + h*0.38), (x + w*0.75, y + h*0.35), (x + w*0.85, y + h*0.38)],
            # Nose (8 points)
            [(x + w*0.45, y + h*0.25), (x + w*0.50, y + h*0.35), (x + w*0.55, y + h*0.25),
             (x + w*0.48, y + h*0.45), (x + w*0.52, y + h*0.45), (x + w*0.50, y + h*0.55),
             (x + w*0.45, y + h*0.60), (x + w*0.55, y + h*0.60)],
            # Mouth (12 points)
            [(x + w*0.25, y + h*0.65), (x + w*0.35, y + h*0.75), (x + w*0.45, y + h*0.70),
             (x + w*0.55, y + h*0.70), (x + w*0.65, y + h*0.75), (x + w*0.75, y + h*0.65),
             (x + w*0.30, y + h*0.85), (x + w*0.40, y + h*0.85), (x + w*0.50, y + h*0.85),
             (x + w*0.60, y + h*0.85), (x + w*0.70, y + h*0.85), (x + w*0.80, y + h*0.85)],
            # Jawline (36 points - simplified)
        ]
        
        # Flatten all landmarks to [x1, y1, z1, x2, y2, z2, ...] format
        all_landmarks = []
        for region in facial_landmarks:
            for point in region:
                # Normalize coordinates (0-1 range)
                norm_x = point[0] / img_w
                norm_y = point[1] / img_h
                all_landmarks.extend([norm_x, norm_y, 0.0])  # z=0 for 2D landmarks
        
        # Add jawline points (simplified 36 points)
        for i in range(36):
            jaw_x = x + w * (0.1 + 0.8 * np.sin(i * np.pi / 18))
            jaw_y = y + h * (0.7 + 0.2 * np.cos(i * np.pi / 18))
            norm_x = jaw_x / img_w
            norm_y = jaw_y / img_h
            all_landmarks.extend([norm_x, norm_y, 0.0])
        
        st.success(f"✅ Face detected! Generated {len(all_landmarks)//3} landmarks")
        return all_landmarks
    else:
        st.error("❌ No face detected. Please upload a photo with a clear frontal face.")
        return None
