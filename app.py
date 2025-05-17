import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO
import zipfile

def remove_light_background(image, threshold=200):
    """Elimina fondos claros y conserva colores oscuros."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(img_array)
    rgba = [b, g, r, mask]
    return Image.fromarray(cv2.merge(rgba))

def process_images(uploaded_files, threshold):
    """Procesa todas las imágenes y devuelve lista de (nombre, imagen)"""
    processed = []
    for file in uploaded_files:
        try:
            img = Image.open(file).convert("RGB")
            processed_img = remove_light_background(img, threshold)
            processed.append((file.name, processed_img))
        except Exception as e:
            st.error(f"Error en {file.name}: {str(e)}")
    return processed

def create_zip(processed_images):
    """Crea un archivo ZIP en memoria con todas las imágenes"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for name, img in processed_images:
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG', quality=100)
            zip_file.writestr(f"processed_{os.path.splitext(name)[0]}.png", img_bytes.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.title("🖼️ Removedor de Fondos + Descarga Masiva")

    # Configuración
    threshold = st.slider("Umbral de detección de fondo claro", 150, 240, 200)
    uploaded_files = st.file_uploader("Sube tus imágenes", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

    if uploaded_files:
        processed_images = process_images(uploaded_files, threshold)

        # Descarga masiva (ZIP)
        if len(processed_images) > 1:
            zip_buffer = create_zip(processed_images)
            st.download_button(
                label="📥 Descargar TODAS las imágenes (ZIP)",
                data=zip_buffer,
                file_name="imagenes_procesadas.zip",
                mime="application/zip"
            )
            st.write("---")

        # Visualización y descarga individual
        for name, img in processed_images:
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(BytesIO(uploaded_files[0].getvalue())), caption="Original", use_column_width=True)
            with col2:
                st.image(img, caption="Procesada", use_column_width=True)

                # Descarga individual
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG', quality=100)
                st.download_button(
                    label=f"⬇️ Descargar {name}",
                    data=img_bytes.getvalue(),
                    file_name=f"processed_{name.split('.')[0]}.png",
                    mime="image/png"
                )
            st.write("---")

if __name__ == "__main__":
    main()
