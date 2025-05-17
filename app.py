import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from io import BytesIO

def remove_light_background(image, threshold=200):
    """Elimina fondos claros y conserva colores oscuros."""
    img_array = np.array(image)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Crear máscara donde los píxeles claros se convierten en transparentes
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Aplicar la máscara al canal alfa
    b, g, r = cv2.split(img_array)
    rgba = [b, g, r, mask]
    result = cv2.merge(rgba)

    return Image.fromarray(result)

def save_image(image, filename, output_folder="output"):
    """Guarda la imagen en PNG con máxima calidad."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
    image.save(output_path, "PNG", quality=100)
    return output_path

def main():
    st.title("Removedor de Fondos Claros")
    st.write("Sube tus ilustraciones para eliminar fondos claros y convertirlas a PNG de alta calidad")

    # Configuración de parámetros
    threshold = st.slider("Umbral de detección de fondo claro", 150, 240, 200)

    # Carga múltiple de archivos
    uploaded_files = st.file_uploader("Sube tus imágenes", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Archivos cargados: {len(uploaded_files)}")
        output_files = []

        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                processed_image = remove_light_background(image, threshold)

                # Mostrar antes/después
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                with col2:
                    st.image(processed_image, caption="Procesada", use_column_width=True)

                # Guardar imagen
                output_path = save_image(processed_image, uploaded_file.name)
                output_files.append(output_path)

                # Botón de descarga
                buf = BytesIO()
                processed_image.save(buf, format="PNG", quality=100)
                st.download_button(
                    label=f"Descargar {uploaded_file.name}",
                    data=buf.getvalue(),
                    file_name=f"processed_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Error procesando {uploaded_file.name}: {str(e)}")

        if output_files:
            st.success(f"Procesamiento completado. Archivos guardados en: {', '.join(output_files)}")

if __name__ == "__main__":
    main()