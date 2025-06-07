import os
import numpy as np
from deepface import DeepFace
import cv2

IMG_SIZE = (160, 160)
THRESHOLD = 0.3  # Adjustable

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def recognize_faces(extracted_dir, ref_dir):
    ref_embeddings = {}
    for ref_img in os.listdir(ref_dir):
        path = os.path.join(ref_dir, ref_img)
        img = load_image(path)
        emb = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)[0]['embedding']
        ref_embeddings[ref_img.split(".")[0]] = emb

    results = {}
    for face_img in os.listdir(extracted_dir):
        face_path = os.path.join(extracted_dir, face_img)
        face_img_data = load_image(face_path)
        face_emb = DeepFace.represent(face_img_data, model_name='Facenet', enforce_detection=False)[0]['embedding']

        best_match = None
        best_score = -1

        for name, ref_emb in ref_embeddings.items():
            score = np.dot(face_emb, ref_emb) / (np.linalg.norm(face_emb) * np.linalg.norm(ref_emb))
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= THRESHOLD:
            results[face_img] = {
                "match": best_match,
                "similarity": float(best_score)
            }
        else:
            results[face_img] = {
                "name": None,
                "similarity": float(best_score)
            }

    return results
