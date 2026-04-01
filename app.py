import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.preprocess import preprocess_image
from src.features import extract_features
from src.matcher import match_features, calculate_score


st.title("🔍 Fingerprint Analyser")

# Upload images
q_file = st.file_uploader("Upload Questioned Fingerprint", type=["jpg", "png"])
a_file = st.file_uploader("Upload Admitted Fingerprint", type=["jpg", "png"])

if q_file and a_file:
    # Convert to OpenCV format
    q_img = np.array(Image.open(q_file).convert("L"))
    a_img = np.array(Image.open(a_file).convert("L"))

    # Preprocess
    img_q = cv2.equalizeHist(q_img)
    img_a = cv2.equalizeHist(a_img)

    # Features
    kp1, desc1 = extract_features(img_q)
    kp2, desc2 = extract_features(img_a)

    # Match
    matches = match_features(desc1, desc2)
    score, good_matches = calculate_score(matches)

    st.subheader(f"Similarity Score: {score:.2f}")

    # Draw matches
    result = cv2.drawMatches(
        img_q, kp1,
        img_a, kp2,
        good_matches[:50],
        None,
        flags=2
    )

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    st.image(result, caption="Fingerprint Matches")
