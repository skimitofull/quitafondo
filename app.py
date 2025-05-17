import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
import zipfile
import tensorflow as tf  # Para el modelo ESRGAN

# Configuración de calidad
PNG_QUALITY = 100
PNG_OPTIMIZE = True
UPSCALE_FACTOR = 4  # Aumento de resolución 4x

# Cargar modelo ESRGAN (pre-entrenado)
@st.cache_resource
def load_esrgan():
    try:
        model = tf.keras.models.load_model('ESRGAN_model.h5')  # Modelo pre-entrenado
        return model
    except:
        st.warning("Modelo ESRGAN no encontrado. Usando interpolación tradicional.")
        return None

def upscale_image(image, model):
    """Aumenta la resolución 4x usando ESRGAN o interpolación"""
    if model:
        # Preprocesamiento para ESRGAN
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        sr_array = model.predict(img_array)[0]
        sr_array = (sr_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(sr_array)
    else:
        # Interpolación tradicional como fallback
        width, height = image.size
        return image.resize(
            (width * UPSCALE_FACTOR, height * UPSCALE_FACTOR),
            resample=Image.Resampling.LANCZOS
        )

def process_image(image, processing_mode, threshold, upscale_model):
    """Procesamiento completo con aumento de resolución"""
    # Paso 1: Procesamiento según modo seleccionado
    if processing_mode == "Fondo claro":
        processed_img = remove_light_background(image, threshold)
    else:
        processed_img = extract_dark_colors(image, 100 - threshold)
        processed_img = processed_img.filter(ImageFilter.SMOOTH_MORE)

    # Paso 2: Aumento de resolución 4x
    sr_img = upscale_image(processed_img, upscale_model)

    return sr_img

def main():
    st.set_page_config(page_title="Super-Resolución 4x", layout="wide")
    st.title("🚀 Aumentador de Resolución 4x + PNG HQ")

    # Cargar modelo (solo una vez)
    esrgan_model = load_esrgan()

    # --- UI ---
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
                "🔵 Umbral de fondo claro:",
                150, 250, 200
            )
        else:
            threshold = st.slider(
                "⚫ Umbral de oscuridad:",
                0, 100, 30
            )

    # --- Carga de archivos ---
    uploaded_files = st.file_uploader(
        "Sube imágenes (PNG/JPG):",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        processed_images = []

        for i, file in enumerate(uploaded_files):
            try:
                original_img = Image.open(file).convert("RGB")

                # Procesamiento completo
                final_img = process_image(original_img, processing_mode, threshold, esrgan_model)
                processed_images.append((file.name, original_img, final_img))

                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Error en {file.name}: {str(e)}")

        # --- Resultados ---
        st.success(f"¡Procesamiento completado! Resolución aumentada 4x")

        # Descarga masiva
        if len(processed_images) > 1:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for name, _, final_img in processed_images:
                    img_bytes = BytesIO()
                    final_img.save(
                        img_bytes,
                        format='PNG',
                        quality=PNG_QUALITY,
                        optimize=PNG_OPTIMIZE,
                        dpi=(600, 600)  # Alta resolución para impresión
                    )
                    zip_file.writestr(
                        f"4X_{name.split('.')[0]}.png",
                        img_bytes.getvalue()
                    )

            st.download_button(
                label="📦 Descargar TODAS (ZIP) - 4x Resolución",
                data=zip_buffer.getvalue(),
                file_name="super_resolution_4x.zip",
                mime="application/zip"
            )

        # Visualización comparativa
        st.subheader("🔍 Comparación Original vs 4x")
        for name, original, final in processed_images:
            with st.expander(f"🖼️ {name}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(original, caption="Original", use_column_width=True)
                    st.text(f"Tamaño: {original.size}")
                with col2:
                    st.image(final, caption="4x Super-Resolución", use_column_width=True)
                    st.text(f"Tamaño: {final.size}")

                    # Descarga individual
                    img_bytes = BytesIO()
                    final.save(img_bytes, format='PNG', quality=100)
                    st.download_button(
                        label=f"⬇️ Descargar {name} (4x)",
                        data=img_bytes.getvalue(),
                        file_name=f"4X_{name}",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
