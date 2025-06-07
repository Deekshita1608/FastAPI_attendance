from ultralytics import YOLO
from super_image import EdsrModel
from PIL import Image
import torch
import numpy as np
import cv2
import os

model = YOLO("weights/yolov8n-face.pt")
sr_model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=4)
sr_model.eval()

def detect_and_save_faces(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape
    results = model(image)
    padding_ratio = 0.5

    for i, (xmin, ymin, xmax, ymax, *_ ) in enumerate(results[0].boxes.data.tolist()):
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        pad_x = int((xmax - xmin) * padding_ratio)
        pad_y = int((ymax - ymin) * padding_ratio)
        xmin, ymin = max(0, xmin - pad_x), max(0, ymin - pad_y)
        xmax, ymax = min(img_width, xmax + pad_x), min(img_height, ymax + pad_y)

        face = image[ymin:ymax, xmin:xmax]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).resize((64, 64))
        face_tensor = torch.from_numpy(np.array(face_pil)).float().permute(2, 0, 1) / 255.0
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            face_sr = sr_model(face_tensor)
        
        sr_img = face_sr.squeeze().permute(1, 2, 0).numpy()
        sr_img = (sr_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"face_{i}.jpg"), cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
