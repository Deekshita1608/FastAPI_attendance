from PIL import Image, ExifTags
from datetime import datetime

def extract_image_datetime(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if not exif_data:
            return None

        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "DateTimeOriginal":
                # Convert "YYYY:MM:DD HH:MM:SS" to datetime object
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

    except Exception as e:
        print("Error reading EXIF datetime:", e)

    return None
