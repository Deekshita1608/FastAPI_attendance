from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from utils.detector import detect_and_save_faces
from utils.recognizer import recognize_faces
from utils.image_utils import extract_image_datetime
from datetime import datetime
import os
import shutil
import uuid

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), date: str = Form(...)):
    """
    Upload an image and validate if its EXIF date matches the selected date (from date picker).
    """
    # Save the uploaded file temporarily
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(temp_dir, unique_filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 1: Extract EXIF datetime from image
    taken_time = extract_image_datetime(file_path)
    timestamp_valid = False

    if taken_time:
        try:
            user_date = datetime.strptime(date, "%Y-%m-%d").date()
            image_date = taken_time.date()
            timestamp_valid = (image_date == user_date)
        except ValueError:
            timestamp_valid=False

    # Step 2: Run face detection and recognition
    face_dir = "extracted_faces"
    detect_and_save_faces(file_path, face_dir)
    results = recognize_faces(face_dir, ref_dir="ref_extracted")
    results["timestamp_valid"] = timestamp_valid

    # Cleanup
    os.remove(file_path)

    return JSONResponse(content=results)
