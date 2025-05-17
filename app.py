import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import zipfile

def remove_light_background(image, threshold):
    """Elimina fondos claros conservando colores oscuros"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    r, g, b = cv2.split(img_array)
    rgba = [r, g, b, mask]
    return Image.fromarray(cv2.merge(rgba))

def extract_dark_colors(image, darkness_threshold):
    """Extrae solo colores oscuros (negros/tonos cercanos)"""
    img_array = np.array(image)
    # Convertir a LAB para mejor detección de oscuridad
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    # Usar el canal L (luminosidad) para detectar oscuridad
    _, mask = cv2.threshold(lab[:,:,0], darkness_threshold, 255, cv2.THRESH_BINARY_INV)
    # Crear imagen transparente
    result = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    result[:,:,:3] = img_array
    result[:,:,3] = mask
    return Image.fromarray(result)

def main():
    st.set_page_config(page_title="Procesador de Ilustraciones", layout="wide")
    st.title("🎨 Procesador Avanzado de Ilustraciones")

    # --- Selector de Modo ---
    processing_mode = st.radio(
        "**MODO DE PROCESAMIENTO:**",
        ("Fondo claro", "Color oscuro"),
        index=0,
        horizontal=True,
        key="mode_selector"
    )

    # --- Parámetros según modo ---
    with st.expander("⚙️ Ajustes avanzados", expanded=True):
        if processing_mode == "Fondo claro":
            threshold = st.slider(
                "Umbral de fondo claro (0-255):",
                100, 250, 200,
                help="Valores altos = elimina más tonos claros"
            )
        else:
            threshold = st.slider(
                "Umbral de oscuridad (0-100):",
                0, 100, 30,
                help="0 = solo negro puro, 100 = incluye grises medios"
            )

    # --- Carga de archivos ---
    uploaded_files = st.file_uploader(
        "Sube tus ilustraciones (PNG/JPG):",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # --- Procesamiento ---
        progress_bar = st.progress(0)
        processed_images = []

        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file).convert("RGB")

                if processing_mode == "Fondo claro":
                    processed_img = remove_light_background(img, threshold)
                else:
                    processed_img = extract_dark_colors(img, 100 - threshold)  # Invertimos el umbral para oscuridad

                processed_images.append((file.name, processed_img))
                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Error procesando {file.name}: {str(e)}")

        # --- Visualización y Descargas ---
        st.success(f"¡Procesamiento completado! ({len(processed_images)} imágenes)")

        # Descarga masiva
        if len(processed_images) > 1:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for name, img in processed_images:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG', quality=100)
                    zip_file.writestr(
                        f"{processing_mode[:3]}_{name.split('.')[0]}.png",
                        img_bytes.getvalue()
                    )
            st.download_button(
                label="📦 Descargar TODAS (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{processing_mode.lower().replace(' ', '_')}.zip",
                mime="application/zip",
                key="mass_download"
            )

        # Galería de resultados
        st.subheader("🔍 Resultados")
        cols = st.columns(2)

        for idx, (name, img) in enumerate(processed_images):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.image(
                        img,
                        caption=f"{name} ({processing_mode})",
                        use_column_width=True
                    )
                    # Descarga individual
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG', quality=100)
                    st.download_button(
                        label=f"⬇️ {name.split('.')[0][:15]}...",
                        data=img_bytes.getvalue(),
                        file_name=f"proc_{name}",
                        mime="image/png",
                        key=f"dl_{idx}"
                    )

if __name__ == "__main__":
    main()
