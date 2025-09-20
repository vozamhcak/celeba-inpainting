import os, io, base64, requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Face Inpainting (CelebA)", layout="wide")

# ---- API ----
default_api = os.getenv("API_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("API URL", default_api)
PREDICT_ENDPOINT = f"{API_URL.rstrip('/')}/predict"
META_ENDPOINT = f"{API_URL.rstrip('/')}/meta"

# Берём целевой размер из API (fallback=64)
try:
    IMG_SIZE = requests.get(META_ENDPOINT, timeout=3).json().get("img_size", 64)
except Exception:
    IMG_SIZE = 64

MAX_AREA_RATIO = 0.30
st.sidebar.caption(f"Модель ожидает {IMG_SIZE}×{IMG_SIZE}. Маска ≤ 30% площади.")
st.title("Face Inpainting")

uploaded = st.file_uploader("Загрузите изображение (JPG/PNG)", type=["png", "jpg", "jpeg"])

def draw_black_rect(img: Image.Image, box):
    m = img.copy()
    ImageDraw.Draw(m).rectangle(box, fill=(0, 0, 0))
    return m

if uploaded:
    # приводим вход к размеру модели
    img_full = Image.open(uploaded).convert("RGB")
    img = img_full.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    display = min(512, IMG_SIZE * 8)
    scale = display / IMG_SIZE
    bg = img.resize((int(IMG_SIZE * scale), int(IMG_SIZE * scale)), Image.NEAREST)

    st.subheader("Нарисуйте прямоугольник (≤ 30% площади)")
    canvas = st_canvas(
        background_image=bg,
        height=int(IMG_SIZE * scale),
        width=int(IMG_SIZE * scale),
        fill_color="rgba(0,0,0,0.4)",
        stroke_color="#E1FF00",
        stroke_width=2,
        drawing_mode="rect",
        update_streamlit=True,
        key="mask_canvas",
    )

    rect_box = None
    area_ratio = 0.0
    if canvas.json_data and canvas.json_data.get("objects"):
        o = canvas.json_data["objects"][-1]
        if o.get("type") == "rect":
            left = o.get("left", 0)
            top = o.get("top", 0)
            width = o.get("width", 0) * o.get("scaleX", 1)
            height = o.get("height", 0) * o.get("scaleY", 1)

            # переводим координаты из дисплейного масштаба -> в 64×64
            x = int(round(left / scale))
            y = int(round(top / scale))
            w = int(round(width / scale))
            h = int(round(height / scale))

            # нормализуем границы
            x = max(0, min(x, IMG_SIZE - 1))
            y = max(0, min(y, IMG_SIZE - 1))
            w = max(1, min(w, IMG_SIZE - x))
            h = max(1, min(h, IMG_SIZE - y))

            rect_box = (x, y, x + w, y + h)
            area_ratio = (w * h) / (IMG_SIZE * IMG_SIZE)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Оригинал (ресайз под модель)**")
        st.image(img, use_column_width=True)

    masked = None
    can_restore = False
    if rect_box:
        masked = draw_black_rect(img, rect_box)  # чистый чёрный (0,0,0)
        with c2:
            st.markdown(f"**Замаскировано** ({area_ratio*100:.2f}%)")
            st.image(masked, use_column_width=True)

        if area_ratio <= MAX_AREA_RATIO:
            can_restore = True
        else:
            st.error("Область > 30% изображения — уменьшите прямоугольник.")

    else:
        st.warning("Нарисуйте прямоугольник на изображении.")

    if st.button("Restore", type="primary", disabled=not can_restore) and masked is not None:
        try:
            buf = io.BytesIO(); masked.save(buf, format="PNG"); buf.seek(0)
            files = {"file": ("masked.png", buf, "image/png")}
            with st.spinner("Восстанавливаем..."):
                r = requests.post(PREDICT_ENDPOINT, files=files, timeout=120)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
            else:
                b64 = r.json().get("result")
                if not b64:
                    st.error("Ответ API без ключа 'result'.")
                else:
                    out = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                    with c3:
                        st.markdown("**Восстановлено**")
                        st.image(out, use_column_width=True)
                    st.success("Маска нарисована")
        except Exception as e:
            st.error(f"Ошибка: {e}")
else:
    st.info("Загрузите изображение, выделите прямоугольник и нажмите Restore.")
