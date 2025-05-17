import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import zipfile

# Configuración de calidad
PNG_QUALITY = 100  # Máxima calidad (0-100)
PNG_OPTIMIZE = True

def remove_light_background(image, threshold):
    """Elimina fondos claros conservando detalles con máxima calidad"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    r, g, b = cv2.split(img_array)
    rgba = [r, g, b, mask]
    return Image.fromarray(cv2.merge(rgba))

def extract_dark_colors(image, darkness_threshold):
    """Extrae colores oscuros preservando bordes nítidos"""
    img_array = np.array(image)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    _, mask = cv2.threshold(lab[:,:,0], darkness_threshold, 255, cv2.THRESH_BINARY_INV)

    # Crear imagen RGBA con máxima calidad
    result = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
    result[:,:,:3] = img_array
    result[:,:,3] = mask

    # Mejorar bordes con filtro Gaussiano
    alpha = cv2.GaussianBlur(result[:,:,3], (3,3), 0)
    result[:,:,3] = alpha

    return Image.fromarray(result)

def save_as_high_quality_png(image):
    """Guarda la imagen en PNG con calidad máxima"""
    img_bytes = BytesIO()
    image.save(
        img_bytes,
        format='PNG',
        quality=PNG_QUALITY,
        optimize=PNG_OPTIMIZE,
        dpi=(300, 300)  # Alta resolución
    )
    img_bytes.seek(0)
    return img_bytes

def main():
    st.set_page_config(page_title="Exportador PNG Profesional", layout="wide")
    st.title("🖼️ Exportador PNG en Máxima Calidad")

    # --- Selector de Modo ---
    col1, col2 = st.columns([1, 3])
    with col1:
        processing_mode = st.radio(
            "**MODO:**",
            ("Fondo claro", "Color oscuro"),
            index=0,
            key="mode_selector"
        )

    with col2:
        if processing_mode == "Fondo claro":
            threshold = st.slider(
                "🔵 Umbral de fondo claro (0-255):",
                150, 250, 200,
                help="Mayor valor = elimina más tonos claros"
            )
        else:
            threshold = st.slider(
                "⚫ Umbral de oscuridad (0-100):",
                0, 100, 30,
                help="0 = negro puro, 100 = incluye grises"
            )

    # --- Carga de archivos ---
    uploaded_files = st.file_uploader(
        "Sube ilustraciones (PNG/JPG):",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Máx. 50 archivos simultáneos"
    )

    if uploaded_files:
        # --- Procesamiento con barra de progreso ---
        progress_bar = st.progress(0)
        processed_images = []

        for i, file in enumerate(uploaded_files):
            try:
                img = Image.open(file).convert("RGB")

                if processing_mode == "Fondo claro":
                    processed_img = remove_light_background(img, threshold)
                else:
                    processed_img = extract_dark_colors(img, 100 - threshold)

                # Aplicar suavizado de bordes para mejor calidad
                processed_img = processed_img.filter(ImageFilter.SMOOTH_MORE) if processing_mode == "Color oscuro" else processed_img

                processed_images.append((file.name, processed_img))
                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Error en {file.name}: {str(e)}")

        # --- Descargas ---
        st.divider()
        st.subheader("💾 Opciones de Exportación PNG")

        # Descarga masiva (ZIP)
        if len(processed_images) > 1:
            with st.expander("📦 DESCARGA MASIVA (ZIP)", expanded=True):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for name, img in processed_images:
                        png_data = save_as_high_quality_png(img)
                        zip_file.writestr(
                            f"HQ_{processing_mode[:3]}_{name.split('.')[0]}.png",
                            png_data.getvalue()
                        )

                st.download_button(
                    label="⬇️ Descargar TODAS (ZIP) - Máxima Calidad",
                    data=zip_buffer.getvalue(),
                    file_name=f"PNG_ULTRAHQ_{processing_mode[:3]}.zip",
                    mime="application/zip",
                    key="mass_download",
                    help="Archivos PNG en calidad 100% sin compresión"
                )

        # Descarga individual
        with st.expander("🖼️ DESCARGA INDIVIDUAL", expanded=True):
            cols = st.columns(2)
            for idx, (name, img) in enumerate(processed_images):
                with cols[idx % 2]:
                    png_data = save_as_high_quality_png(img)
                    st.download_button(
                        label=f"⬇️ {name[:25]}... (PNG HQ)",
                        data=png_data.getvalue(),
                        file_name=f"HQ_{name.split('.')[0]}.png",
                        mime="image/png",
                        key=f"dl_{idx}",
                        help="PNG en calidad 100%"
                    )
                    st.image(img, use_column_width=True)

if __name__ == "__main__":
    main()
