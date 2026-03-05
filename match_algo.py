import os
import pickle
import json
import traceback
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action="ignore")

from pages.helper import db_queries

def get_public_cases_data(status="NF"):
    try:
        result = db_queries.fetch_public_cases(train_data=True, status=status)
        d1 = pd.DataFrame(result, columns=["label", "face_mesh"])
        d1["face_mesh"] = d1["face_mesh"].apply(lambda x: json.loads(x))
        d2 = pd.DataFrame(d1.pop("face_mesh").values.tolist(), index=d1.index).rename(
            columns=lambda x: "fm_{}".format(x + 1)
        )
        df = d1.join(d2)
        
        # 🔥 FIX: Handle NaN values during numeric conversion
        for col in df.columns:
            if col != "label":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Fill NaN with column mean
                df[col] = df[col].fillna(df[col].mean())
        
        # Final cleanup - ensure no NaN remains
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df
    except Exception as e:
        traceback.print_exc()
        return None

def get_registered_cases_data(status="NF"):
    try:
        from pages.helper.db_queries import engine, RegisteredCases
        import pandas as pd
        import json
        from sqlmodel import Session, select

        with Session(engine) as session:
            result = session.exec(
                select(
                    RegisteredCases.id,
                    RegisteredCases.face_mesh,
                    RegisteredCases.status,
                )
            ).all()
            d1 = pd.DataFrame(result, columns=["label", "face_mesh", "status"])
            if status:
                d1 = d1[d1["status"] == status]
            d1["face_mesh"] = d1["face_mesh"].apply(lambda x: json.loads(x))
            d2 = pd.DataFrame(
                d1.pop("face_mesh").values.tolist(), index=d1.index
            ).rename(columns=lambda x: "fm_{}".format(x + 1))
            df = d1.join(d2)
            
            # 🔥 FIX: Handle NaN values during numeric conversion
            for col in df.columns:
                if col not in ["label", "status"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    # Fill NaN with column mean
                    df[col] = df[col].fillna(df[col].mean())
            
            # Final cleanup - ensure no NaN remains
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            return df
    except Exception as e:
        traceback.print_exc()
        return None

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def match(distance_threshold=3):
    matched_images = defaultdict(list)
    public_cases_df = get_public_cases_data()
    registered_cases_df = get_registered_cases_data()

    if public_cases_df is None or registered_cases_df is None:
        return {"status": False, "message": "Couldn't connect to database"}
    if len(public_cases_df) == 0 or len(registered_cases_df) == 0:
        return {"status": False, "message": "No public or registered cases found"}

    # Store original labels before encoding
    original_reg_labels = registered_cases_df.iloc[:, 0].tolist()
    original_pub_labels = public_cases_df.iloc[:, 0].tolist()

    # 🔥 FIX: Prepare features with NaN handling
    reg_features = registered_cases_df.iloc[:, 2:].values.astype(float)
    
    # Replace any remaining NaN/inf with mean values
    imputer = SimpleImputer(strategy='mean')
    reg_features = imputer.fit_transform(reg_features)
    
    # Ensure no inf values
    reg_features = np.nan_to_num(reg_features, nan=0.0, posinf=1.0, neginf=-1.0)

    # Pad/truncate features to same length if needed
    max_len = reg_features.shape[1]
    if reg_features.shape[1] < max_len:
        reg_features = np.pad(reg_features, ((0,0),(0, max_len - reg_features.shape[1])), 'constant')

    # Create simple numeric labels for KNN (0, 1, 2, ...)
    numeric_labels = list(range(len(reg_features)))

    # Scale features for better KNN performance
    scaler = StandardScaler()
    reg_features_scaled = scaler.fit_transform(reg_features)

    # Train KNN classifier with numeric labels - NOW NaN SAFE
    knn = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree", weights="distance")
    knn.fit(reg_features_scaled, numeric_labels)

    # Process public cases
    for i, row in public_cases_df.iterrows():
        pub_label = original_pub_labels[i]
        face_encoding = np.array(row[2:].values).astype(float)  # Skip label column
        
        try:
            # 🔥 FIX: Clean public features same way
            face_encoding = imputer.transform([face_encoding])[0]
            face_encoding = np.nan_to_num(face_encoding, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Pad if needed
            if len(face_encoding) < max_len:
                face_encoding = np.pad(face_encoding, (0, max_len - len(face_encoding)), 'constant')
            
            # Scale
            face_encoding_scaled = scaler.transform([face_encoding])[0]
            
            # Get distances to nearest neighbors
            distances, indices = knn.kneighbors([face_encoding_scaled])
            closest_distance = distances[0][0]
            
            print(f"Distance for case {pub_label}: {closest_distance}")

            # Check if distance meets threshold criteria
            if closest_distance < distance_threshold:  # Lower distance = better match
                predicted_idx = indices[0][0]
                reg_label = original_reg_labels[predicted_idx]
                matched_images[reg_label].append(pub_label)
                print(f"✅ MATCH: {pub_label} -> {reg_label} (dist: {closest_distance:.2f})")
            else:
                print(f"❌ No match for {pub_label} (dist: {closest_distance:.2f} > {distance_threshold})")
                
        except Exception as e:
            print(f"Error processing public case {pub_label}: {str(e)}")
            continue

    print(f"🎯 Final matches found: {len(matched_images)}")
    return {"status": True, "result": dict(matched_images)}

if __name__ == "__main__":
    result = match()
    print(result)
