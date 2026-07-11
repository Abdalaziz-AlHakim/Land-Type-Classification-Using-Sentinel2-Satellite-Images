import keras
 
def _strip_axes_kwargs(cls):
    original_init = cls.__init__
 
    def patched_init(self, *args, **kwargs):
        kwargs.pop("input_axes", None)
        kwargs.pop("output_axes", None)
        original_init(self, *args, **kwargs)
 
    cls.__init__ = patched_init
 
 
_strip_axes_kwargs(keras.initializers.VarianceScaling)
for _sub in keras.initializers.VarianceScaling.__subclasses__():
    _strip_axes_kwargs(_sub)
 
_original_bn_init = keras.layers.BatchNormalization.__init__
 
def _patched_bn_init(self, *args, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    kwargs.pop("synchronized", None)
    _original_bn_init(self, *args, **kwargs)
 
keras.layers.BatchNormalization.__init__ = _patched_bn_init
 
_original_dense_init = keras.layers.Dense.__init__
 
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)
 
keras.layers.Dense.__init__ = _patched_dense_init
 
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import os
from streamlit_folium import st_folium
import folium
import requests
import io
import rasterio


"""
sentinel_functions.py
Fetches a real Sentinel-2 satellite image from the Copernicus Data Space
(Sentinel Hub API) for a given latitude/longitude.
"""

import requests
from PIL import Image
import io
import streamlit as st


def get_sentinelhub_token():
    """
    Gets an access token using Client ID and Client Secret.
    These should be stored in st.secrets (never hardcoded in the code).
    """
    client_id = os.environ.get("SH_CLIENT_ID")
    client_secret = os.environ.get("SH_CLIENT_SECRET")

    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    response = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]
    
@st.cache_data(ttl=3600, show_spinner=".")

def fetch_sentinel2_image(lat, lon, delta=0.02, size=400):
    """
    Fetches a real Sentinel-2 image around a (lat, lon) point.
    delta: size of the area around the point (in degrees)
    size: image dimensions in pixels (square)
    """
    token = get_sentinelhub_token()

    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B02", "B03", "B04"],
        output: { bands: 3 }
      };
    }
    function evaluatePixel(sample) {
      return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {"maxCloudCoverage": 30},
                }
            ],
        },
        "output": {
            "width": size,
            "height": size,
            "responses": [
                {"identifier": "default", "format": {"type": "image/png"}}
            ],
        },
        "evalscript": evalscript,
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(process_url, json=payload, headers=headers)
    response.raise_for_status()

    return Image.open(io.BytesIO(response.content)).convert("RGB")
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sentinel-2 Land Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ============================================================
# CLASS METADATA (icon + color per land type)
# ============================================================
CLASS_INFO = {
    "AnnualCrop":           {"icon": "🌾", "color": "#D4A017", "desc": "Seasonal cropland"},
    "Forest":               {"icon": "🌲", "color": "#1B7A3D", "desc": "Dense forest cover"},
    "HerbaceousVegetation": {"icon": "🌿", "color": "#4CAF50", "desc": "Grass & shrub vegetation"},
    "Highway":              {"icon": "🛣️", "color": "#6B7280", "desc": "Roads & highways"},
    "Industrial":           {"icon": "🏭", "color": "#8B5E3C", "desc": "Industrial buildings"},
    "Pasture":              {"icon": "🐄", "color": "#8BC34A", "desc": "Grazing pastureland"},
    "PermanentCrop":        {"icon": "🍇", "color": "#A0522D", "desc": "Orchards & vineyards"},
    "Residential":          {"icon": "🏘️", "color": "#E07A5F", "desc": "Urban residential area"},
    "River":                {"icon": "🌊", "color": "#1E88E5", "desc": "River / waterway"},
    "SeaLake":              {"icon": "🏞️", "color": "#0D47A1", "desc": "Sea or lake body"},
}
CLASS_NAMES = list(CLASS_INFO.keys())
 
# ============================================================
# GLOBAL STYLES
# ============================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
 
    .block-container {
        padding-top: 1.2rem;
        max-width: 1150px;
    }
 
    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #1E1B4B 0%, #4C1D95 55%, #7C3AED 100%);
        border-radius: 18px;
        padding: 38px 40px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(76, 29, 149, 0.28);
    }
    .hero h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin: 0;
    }
 
    /* Card container */
    .card {
        background: white;
        border-radius: 16px;
        padding: 22px 26px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 1px solid #EEF2F1;
        margin-bottom: 18px;
    }
 
    /* Prediction headline */
    .pred-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: #F5F3FF;
        border: 1px solid #DDD6FE;
        color: #4C1D95;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 10px 18px;
        border-radius: 12px;
        margin-bottom: 14px;
    }
 
    .conf-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
    }
 
    /* Probability bar row */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 9px;
    }
    .prob-label {
        width: 175px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #374151;
        white-space: nowrap;
    }
    .prob-track {
        flex: 1;
        background: #F1F5F4;
        border-radius: 8px;
        height: 16px;
        overflow: hidden;
        position: relative;
    }
    .prob-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.4s ease;
    }
    .prob-value {
        width: 52px;
        text-align: right;
        font-size: 0.82rem;
        color: #6B7280;
        font-variant-numeric: tabular-nums;
    }
 
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #FAF8FF;
    }
    .sidebar-card {
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid #EDE9FE;
        margin-top: 10px;
    }
 
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
 
# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🛰️ Sentinel-2 Land Type Classifier</h1>
    <p>Upload a satellite image, or drop a pin on the map — our EfficientNetV2-S model
    (GeM pooling, trained on EuroSAT) will identify the land cover type in seconds.</p>
</div>
""", unsafe_allow_html=True)
 
 
# ============================================================
# MODEL LOADING
# ============================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class GeMPooling(layers.Layer):
    def __init__(self, p_init=3.0, p_trainable=True, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.p_init       = p_init
        self.p_trainable  = p_trainable
        self.eps          = eps
 
    def build(self, input_shape):
        self.p = self.add_weight(
            name="gem_p",
            shape=(1,),
            initializer=tf.constant_initializer(self.p_init),
            trainable=self.p_trainable,
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg()
        )
        super().build(input_shape)
 
    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        x = tf.clip_by_value(x, self.eps, tf.reduce_max(x))
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tf.pow(x, 1.0 / self.p)
        return x
 
    def get_config(self):
        config = super().get_config()
        config.update({
            "p_init": self.p_init,
            "p_trainable": self.p_trainable,
            "eps": self.eps
        })
        return config
 
 
def find_available_models():
    model_paths = []
    if os.path.exists("outputs"):
        for f in os.listdir("outputs"):
            if f.endswith((".keras", ".h5")):
                model_paths.append(os.path.join("outputs", f))
    for f in os.listdir("."):
        if f.endswith((".keras", ".h5")):
            model_paths.append(f)
 
    defaults = [
        "outputs/rgb_efficientnetv2s_final.keras",
        "rgb_efficientnetv2s_final.keras",
        "outputs/rgb_classification_model.keras",
        "rgb_classification_model.keras",
        "EuroSAT_MobileNetV2_Final.keras"
    ]
    for d in defaults:
        if d not in model_paths:
            model_paths.append(d)
    return model_paths
 
 
@st.cache_resource
def load_trained_model(path):
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(
                path,
                custom_objects={"GeMPooling": GeMPooling},
                compile=False
            )
            return model
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {str(e)}")
            return None
    return None
 
 
# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    available_models = find_available_models()
    existing_models = [m for m in available_models if os.path.exists(m)]
 
    if existing_models:
        selected_model_path = st.selectbox(
            "Select model file",
            options=existing_models,
            index=0
        )
        model = load_trained_model(selected_model_path)
        loaded_path = selected_model_path
        st.success("Model loaded ✅")
    else:
        selected_model_path = st.selectbox(
            "Select model file (none found)",
            options=available_models,
            index=0
        )
        model = None
        loaded_path = None
        st.warning("⚠️ No model file (.keras or .h5) found.")
        st.caption("Place your trained model in the root directory or in `outputs/`.")
 
    st.markdown("""
    <div class="sidebar-card">
        <b>🧠 About the model</b><br><br>
        <b>Architecture:</b> EfficientNetV2-S<br>
        <b>Pooling:</b> Generalized Mean (GeM)<br>
        <b>Dataset:</b> EuroSAT (RGB)<br>
        <b>Classes:</b> 10 land-use categories
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🗂️ Land type legend**")
    legend_html = ""
    for name, info in CLASS_INFO.items():
        legend_html += f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <span style="width:10px;height:10px;border-radius:50%;background:{info['color']};display:inline-block;"></span>
            <span style="font-size:0.85rem;">{info['icon']} {name}</span>
        </div>"""
    st.markdown(legend_html, unsafe_allow_html=True)
 
 
# ============================================================
# MULTISPECTRAL TIFF HELPERS
# ============================================================
def read_tiff_bands(file_bytes):
    """Open a (multi-band) GeoTIFF with rasterio and return the raw band stack."""
    with rasterio.open(io.BytesIO(file_bytes)) as src:
        arr = src.read()  # shape: (bands, H, W)
    return arr  # float/int array, bands-first
 
 
def stretch_band(band, low=2, high=98):
    """Percentile contrast stretch a single band to 0-255 uint8."""
    band = band.astype(np.float32)
    p_low, p_high = np.percentile(band, (low, high))
    if p_high - p_low < 1e-6:
        p_high = p_low + 1e-6
    band = np.clip((band - p_low) / (p_high - p_low), 0, 1) * 255.0
    return band.astype(np.uint8)
 
 
def compose_rgb_from_bands(band_stack, r_idx, g_idx, b_idx):
    """Build a stretched RGB PIL image from chosen band indices (1-based, rasterio style)."""
    r = stretch_band(band_stack[r_idx - 1])
    g = stretch_band(band_stack[g_idx - 1])
    b = stretch_band(band_stack[b_idx - 1])
    rgb = np.dstack([r, g, b])
    return Image.fromarray(rgb, mode="RGB")
 
 
# ============================================================
# INPUT: TABS (Upload / Map)
# ============================================================
tab1, tab2 = st.tabs(["📤  Upload Image", "🗺️  Select from Map"])
 
selected_image = None
map_data = None
 
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
 
    image_mode = st.radio(
        "Image type",
        options=["RGB Image (JPEG/PNG)", "Multispectral Image (TIFF)"],
        horizontal=True,
        label_visibility="collapsed"
    )
 
    if image_mode == "RGB Image (JPEG/PNG)":
        uploaded_file = st.file_uploader(
            "Choose a satellite image",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
        else:
            st.caption("Supported formats: JPG, PNG, JPEG")
 
    else:
        uploaded_file = st.file_uploader(
            "Choose a multispectral GeoTIFF",
            type=["tif", "tiff"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                band_stack = read_tiff_bands(file_bytes)
                n_bands = band_stack.shape[0]
                st.success(f"✅ File loaded — {n_bands} band(s) detected.")
 
                if n_bands < 3:
                    st.error("This file has fewer than 3 bands — at least 3 are needed to build an RGB composite.")
                else:
                    st.caption("Pick which band goes into each color channel:")
                    band_options = list(range(1, n_bands + 1))
 
                    # sensible defaults for typical Sentinel-2 stacks (B4=Red, B3=Green, B2=Blue)
                    default_r = 4 if n_bands >= 4 else 3
                    default_g = 3 if n_bands >= 3 else 2
                    default_b = 2 if n_bands >= 2 else 1
 
                    bc1, bc2, bc3 = st.columns(3)
                    with bc1:
                        r_idx = st.selectbox("🔴 Red channel ← band", band_options,
                                              index=band_options.index(default_r) if default_r in band_options else 0)
                    with bc2:
                        g_idx = st.selectbox("🟢 Green channel ← band", band_options,
                                              index=band_options.index(default_g) if default_g in band_options else 0)
                    with bc3:
                        b_idx = st.selectbox("🔵 Blue channel ← band", band_options,
                                              index=band_options.index(default_b) if default_b in band_options else 0)
 
                    selected_image = compose_rgb_from_bands(band_stack, r_idx, g_idx, b_idx)
                    st.caption("A 2–98 percentile contrast stretch is applied automatically to each channel for a clear, well-exposed composite.")
            except Exception as e:
                st.error(f"Could not read this TIFF file: {str(e)}")
        else:
            st.caption("Upload a multi-band GeoTIFF (e.g. Sentinel-2 stack) — you'll choose which bands map to R, G, B.")
 
    st.markdown('</div>', unsafe_allow_html=True)
 
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Click anywhere on the map to select a location")
 
    m = folium.Map(location=[30.0444, 31.2357], zoom_start=6, tiles=None)
 
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False
    ).add_to(m)
 
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Labels",
        name="Labels",
        overlay=True
    ).add_to(m)
 
    map_data = st_folium(m, width=None, height=480, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
 
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        st.markdown(
            f'<div class="pred-badge" style="font-size:1rem;">📍 Selected location: '
            f'{lat:.4f}, {lon:.4f}</div>',
            unsafe_allow_html=True
        )
 
        with st.spinner("Fetching real Sentinel-2 image for this location..."):
            try:
                selected_image = fetch_sentinel2_image(lat, lon)
                st.caption("✅ Real Sentinel-2 imagery — matches the data the model was trained on.")
            except Exception as e:
                st.warning("Sentinel-2 fetch failed, falling back to Esri preview.")
                delta = 0.02
                bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
                esri_url = (
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/export"
                    f"?bbox={bbox}&bboxSR=4326&size=400,400&format=png&f=image"
                )
                response = requests.get(esri_url)
                if response.status_code == 200:
                     selected_image = Image.open(io.BytesIO(response.content)).convert("RGB")
 
 
# ============================================================
# RESULTS
# ============================================================
def confidence_color(conf):
    if conf >= 80:
        return "#10B981"   # green
    elif conf >= 50:
        return "#F59E0B"   # amber
    else:
        return "#EF4444"   # red
 
 
if selected_image is not None:
    col1, col2 = st.columns([1, 1.15], gap="large")
 
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🖼️ Input Image")
        st.image(selected_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Prediction Results")
 
        if model is None:
            st.error("Cannot perform classification. No model is loaded.")
        else:
            with st.spinner("Classifying image..."):
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
                img_resized = selected_image.resize((224, 224), Image.BILINEAR)
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array)[0]
                predicted_class_idx = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                info = CLASS_INFO[predicted_class]
                confidence = predictions[predicted_class_idx] * 100
                c_color = confidence_color(confidence)
 
                st.markdown(f"""
                <div class="pred-badge">
                    {info['icon']} {predicted_class}
                    <span class="conf-pill" style="background:{c_color};">{confidence:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.caption(info["desc"])
 
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Probability breakdown**")
 
                order = np.argsort(predictions)[::-1]
                for idx in order:
                    name = CLASS_NAMES[idx]
                    prob = float(predictions[idx]) * 100
                    color = CLASS_INFO[name]["color"]
                    icon = CLASS_INFO[name]["icon"]
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-label">{icon} {name}</div>
                        <div class="prob-track">
                            <div class="prob-fill" style="width:{prob}%; background:{color};"></div>
                        </div>
                        <div class="prob-value">{prob:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
 
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info(" Upload a satellite image or pick a spot on the map to see classification results.")
