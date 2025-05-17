import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO
import zipfile

def remove_light_background(image, threshold):
    """Elimina fondos claros conservando colores oscuros"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(img_array)
    rgba = [b, g, r, mask]
    return Image.fromarray(cv2.merge(rgba))

def extract_dark_colors(image, darkness_threshold=50):
    """Extrae solo colores oscuros (negros/tonos cercanos)"""
    img_array = np.array(image)
    # Convertir a LAB para mejor detección de oscuridad
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    _, a, b = cv2.split(lab)
    # Usar el canal L (luminosidad) para detectar oscuridad
    _, mask = cv2.threshold(lab[:,:,0], darkness_threshold, 255, cv2.THRESH_BINARY_INV)
    # Aplicar máscara
    result = cv2.bitwise_and(img_array, img_array, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
    result[:,:,3] = mask
    return Image.fromarray(result)

def process_images(uploaded_files, processing_mode, threshold):
    processed = []
    for file in uploaded_files:
        try:
            img = Image.open(file).convert("RGB")
            if processing_mode == "Fondo claro":
                processed_img = remove_light_background(img, threshold)
            else:  # Modo "Color oscuro"
                processed_img = extract_dark_colors(img, threshold)
            processed.append((file.name, processed_img))
        except Exception as e:
            st.error(f"Error en {file.name}: {str(e)}")
    return processed

def create_zip(processed_images):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for name, img in processed_images:
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG', quality=100)
            zip_file.writestr(f"processed_{os.path.splitext(name)[0]}.png", img_bytes.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.title("🎨 Procesador Avanzado de Ilustraciones")

    # Selector de modo
    processing_mode = st.radio(
        "Selecciona el modo de procesamiento:",
        ("Fondo claro", "Color oscuro"),
        horizontal=True
    )

    # Configuración dinámica según modo
    if processing_mode == "Fondo claro":
        threshold = st.slider("Umbral de detección de fondo claro", 150, 240, 200,
                            help="Ajusta para eliminar fondos claros conservando detalles oscuros")
    else:
        threshold = st.slider("Umbral de oscuridad", 10, 100, 50,
                            help="Valores más altos = solo colores más oscuros (negro puro = 0)")

    uploaded_files = st.file_uploader("Sube tus ilustraciones", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

    if uploaded_files:
        processed_images = process_images(uploaded_files, processing_mode, threshold)

        # Descarga masiva
        if len(processed_images) > 1:
            zip_buffer = create_zip(processed_images)
            st.download_button(
                label="📦 Descargar TODAS (ZIP)",
                data=zip_buffer,
                file_name=f"ilustraciones_{processing_mode.lower().replace(' ', '_')}.zip",
                mime="application/zip"
            )
            st.write("---")

        # Visualización individual
        for idx, (name, img) in enumerate(processed_images):
            st.subheader(f"🖼️ {name}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(BytesIO(uploaded_files[idx].getvalue())),
                        caption="Original", use_column_width=True)
            with col2:
                st.image(img, caption=f"Procesada ({processing_mode})", use_column_width=True)

                # Descarga individual
                img_bytes = BytesIO()
                img.save(img_bytes, format='PNG', quality=100)
                st.download_button(
                    label=f"⬇️ Descargar {name}",
                    data=img_bytes.getvalue(),
                    file_name=f"{processing_mode.lower().replace(' ', '_')}_{name}",
                    mime="image/png"
                )
            st.write("---")

if __name__ == "__main__":
    main()
