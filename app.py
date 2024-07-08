import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
from utils import extract_plate_text_easy_ocr, visualize_plate
import json

model = YOLO('models\\detect\\train\\weights\\best.pt')


def main():
    st.title('VietNam License Plate Detection & Recognization')
    file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    if file is not None:
        col1, col2 = st.columns(2)

        image = Image.open(file)
        img_path = 'upload_img.jpg'
        image.save(img_path)

        prediction = model.predict(img_path)
        prediction = json.loads(prediction[0].tojson())
        visualize_img, plate = visualize_plate(img_path, prediction)
        st.write(plate)
        with col1:
            st.image(image, 'Uploaded Image')
        with col2:
            st.image(visualize_img, 'Processed Image')


if __name__ == "__main__":
    main()
