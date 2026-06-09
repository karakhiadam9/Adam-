"""
PFE — Classification intelligente d'emails FR / AR
Catégories : Facturation | Technique | RH | Réclamation
Analyse Automatisée Emails + Images + Fichiers CSV
"""

# ══════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════
import re
import io
import csv
import datetime
import shutil

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import cv2  # pip install opencv-python-headless

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


# ══════════════════════════════════════════════════════════
# CONFIG STREAMLIT
# ══════════════════════════════════════════════════════════
st.set_page_config(page_title="PFE - Classification Emails", page_icon="📧", layout="wide")


# ══════════════════════════════════════════════════════════
# CONFIGURATION TESSERACT
# ══════════════════════════════════════════════════════════
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Formats d'images acceptés
ACCEPTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif"]


# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 0.40

CSV_COLUMNS = [
    "Horodatage",
    "Email",
    "Categorie",
    "Confiance (%)",
    "Fiabilite",
    "Langue"
]


# ══════════════════════════════════════════════════════════
# DATASET D'ENTRAINEMENT
# ══════════════════════════════════════════════════════════
DATASET = [
    # FACTURATION
    ("Ma facture est incorrecte.", "Facturation"),
    ("Je souhaite un remboursement.", "Facturation"),
    ("Pouvez-vous envoyer ma facture ?", "Facturation"),
    ("فاتورتي غير صحيحة.", "Facturation"),
    ("أريد استرداد المبلغ.", "Facturation"),

    # TECHNIQUE
    ("Je ne peux pas me connecter.", "Technique"),
    ("Erreur sur le site.", "Technique"),
    ("Le serveur ne fonctionne pas.", "Technique"),
    ("لا أستطيع تسجيل الدخول.", "Technique"),
    ("هناك خطأ في الموقع.", "Technique"),

    # RH
    ("Je veux poser un conge.", "RH"),
    ("Je n'ai pas recu mon salaire.", "RH"),
    ("أريد طلب إجازة.", "RH"),
    ("لم أتلق راتبي.", "RH"),

    # RECLAMATION
    ("Je suis mecontent du service.", "Reclamation"),
    ("Service catastrophique.", "Reclamation"),
    ("الخدمة سيئة جداً.", "Reclamation"),
    ("أريد تقديم شكوى.", "Reclamation"),
]


# ══════════════════════════════════════════════════════════
# PREPROCESSING TEXTE
# ══════════════════════════════════════════════════════════
def normalize_arabic(text):
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text


def preprocess(text):
    text = str(text).lower()
    text = normalize_arabic(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text):
    if re.search(r"[\u0600-\u06FF]", str(text)):
        return "Arabe"
    return "Francais"


# ══════════════════════════════════════════════════════════
# OCR / IMAGE PROCESSING
# ══════════════════════════════════════════════════════════
def pil_to_cv2(pil_img):
    img_array = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def auto_rotate_image(pil_img):
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    return pil_img


def deskew_image(cv2_img):
    try:
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return cv2_img

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 0.5:
            h, w = cv2_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cv2_img = cv2.warpAffine(
                cv2_img,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
    except Exception:
        pass
    return cv2_img


def enhance_image_for_ocr(pil_img):
    pil_img = auto_rotate_image(pil_img)
    pil_img = pil_img.convert("RGB")

    w, h = pil_img.size
    if w < 1000 and w > 0:
        scale = 1000 / w
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(2.0)

    enhancer_sharp = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer_sharp.enhance(2.0)

    cv2_img = pil_to_cv2(pil_img)
    cv2_img = deskew_image(cv2_img)

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    return Image.fromarray(binary)


def detect_image_type(pil_img):
    w, h = pil_img.size
    if h == 0:
        return "document"
    aspect = w / h
    if aspect > 1.5 or aspect < 0.5:
        return "screenshot"
    return "document"


def ocr_image(pil_img):
    preprocessed = enhance_image_for_ocr(pil_img)
    img_type = detect_image_type(pil_img)

    if img_type == "screenshot":
        config = "--oem 3 --psm 6"
    else:
        config = "--oem 3 --psm 3"

    try:
        text = pytesseract.image_to_string(preprocessed, lang="fra+ara", config=config)
        if len(text.strip()) > 10:
            return text, preprocessed, "fra+ara"
    except Exception:
        pass

    try:
        text = pytesseract.image_to_string(preprocessed, lang="fra", config=config)
        if len(text.strip()) > 10:
            return text, preprocessed, "fra"
    except Exception:
        pass

    try:
        text = pytesseract.image_to_string(preprocessed, lang="ara", config=config)
        if len(text.strip()) > 10:
            return text, preprocessed, "ara"
    except Exception:
        pass

    try:
        text = pytesseract.image_to_string(preprocessed, config=config)
        return text, preprocessed, "auto"
    except Exception as e:
        return "", preprocessed, f"erreur: {e}"


# ══════════════════════════════════════════════════════════
# MODELE IA
# ══════════════════════════════════════════════════════════
@st.cache_resource
def train_model():
    texts = [preprocess(t) for t, _ in DATASET]
    labels = [l for _, l in DATASET]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 6),
            max_features=80000,
            sublinear_tf=True
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000, C=1.2),
            cv=2
        ))
    ])
    pipeline.fit(texts, labels)
    return pipeline


def classify(text, model):
    clean = preprocess(text)
    pred = model.predict([clean])[0]
    probs = model.predict_proba([clean])[0]
    scores = dict(zip(model.classes_, probs))
    conf = float(scores[pred])

    if conf >= 0.75:
        fiabilite = "Elevee"
    elif conf >= 0.50:
        fiabilite = "Moyenne"
    elif conf >= CONFIDENCE_THRESHOLD:
        fiabilite = "Faible"
    else:
        fiabilite = "Tres faible"

    return {
        "pred": pred,
        "confidence": conf,
        "scores": scores,
        "langue": detect_language(text),
        "fiabilite": fiabilite
    }


# ══════════════════════════════════════════════════════════
# GESTION HISTORIQUE ET EXPORT
# ══════════════════════════════════════════════════════════
def push_history(email_text, result):
    st.session_state.history.append({
        "Horodatage": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Email": str(email_text)[:80],
        "Categorie": result["pred"],
        "Confiance (%)": round(result["confidence"] * 100, 1),
        "Fiabilite": result["fiabilite"],
        "Langue": result["langue"],
    })


def build_csv(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=CSV_COLUMNS,
        delimiter=";",
        quoting=csv.QUOTE_ALL
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    csv_text = "\ufeff" + buf.getvalue()
    return csv_text.encode("utf-8")


# ══════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════
def fig_scores(scores):
    cats = sorted(scores, key=lambda c: -scores[c])
    vals = [scores[c] for c in cats]
    colors = ["#2ecc71" if v == max(vals) else "#bdc3c7" for v in vals]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(cats, vals, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title("Probabilités par catégorie")
    plt.tight_layout()
    return fig


def fig_history(history):
    df = pd.DataFrame(history)
    if not df.empty:
        counts = df["Categorie"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax, color="#3498db")
        ax.set_title("Répartition des analyses")
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#ffffff")
        plt.tight_layout()
        return fig
    return None


# ══════════════════════════════════════════════════════════
# NER — EXTRACTION D'ENTITÉS
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        return spacy.load("fr_core_news_sm")
    except Exception:
        return None


def extract_entities(text):
    entities = []
    nlp = load_spacy_model()

    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PER", "ORG", "LOC", "DATE", "MISC"]:
                    label_map = {
                        "PER": "Personne",
                        "ORG": "Organisation/Entreprise",
                        "LOC": "Lieu",
                        "DATE": "Date/Heure",
                        "MISC": "Référence/Divers"
                    }
                    entities.append({
                        "Entité": ent.text,
                        "Type": label_map.get(ent.label_, ent.label_)
                    })
        except Exception:
            pass

    invoice_match = re.findall(
        r"(?:facture|n°|numéro)\s*[:#-]?\s*([A-Z0-9-]+)",
        text,
        re.IGNORECASE
    )
    for num in invoice_match:
        entities.append({"Entité": num, "Type": "Numéro de Facture (Détecté)"})

    return entities


def display_ner_section(text):
    st.subheader("🔍 Extraction d'informations clés (NER)")
    with st.spinner("Extraction des entités en cours..."):
        data = extract_entities(text)
        if data:
            df_ner = pd.DataFrame(data).drop_duplicates()
            st.table(df_ner)
        else:
            st.info("Aucune entité spécifique détectée ou modèle spaCy indisponible.")


# ══════════════════════════════════════════════════════════
# STYLE CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── FOND GLOBAL ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf0;
}

[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2640;
}

/* ── HEADER PRINCIPAL ── */
.main-header {
    background: linear-gradient(135deg, #0d1120 0%, #111827 50%, #0a0e1a 100%);
    border: 1px solid #1e2d5e;
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #6b7ab8;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #60a5fa;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── ONGLETS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0d1120;
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
    border: 1px solid #1e2640;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7ab8 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 10px 20px !important;
    transition: all 0.2s ease;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
}

/* ── CARDS ── */
.card {
    background: #0d1120;
    border: 1px solid #1e2640;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
}

/* ── ZONES DE TEXTE ── */
[data-testid="stTextArea"] textarea {
    background: #111827 !important;
    border: 1px solid #1e2d5e !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── BOUTONS ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(37,99,235,0.45) !important;
}

/* ── MÉTRIQUES ── */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e2640;
    border-radius: 14px;
    padding: 20px !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7ab8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 2px dashed #1e2d5e !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2640 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── INFO / WARNING / ERROR ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left: 4px solid #3b82f6 !important;
    background: rgba(59,130,246,0.08) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1e2640 !important;
    border-radius: 12px !important;
}

/* ── SELECT / RADIO ── */
[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label {
    color: #a0aec0 !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #1d4ed8, #60a5fa) !important;
    border-radius: 10px !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d5e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* ── SUBHEADERS ── */
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}
</style>

<div class="main-header">
    <div class="badge">PFE · IA · NLP</div>
    <h1>Classification intelligente d'emails</h1>
    <p>Analyse automatique en Français et Arabe · Facturation · Technique · RH · Réclamation</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []

if "user_info" not in st.session_state:
    st.session_state.user_info = None


# ══════════════════════════════════════════════════════════
# PAGE DE GARDE
# ══════════════════════════════════════════════════════════
def show_landing_page():
    st.markdown("""
    <style>
    .landing-container {
        max-width: 520px;
        margin: 40px auto;
        background: #0d1120;
        border: 1px solid #1e2d5e;
        border-radius: 24px;
        padding: 50px 45px;
        box-shadow: 0 30px 80px rgba(0,0,0,0.5);
    }
    .landing-logo {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 6px;
    }
    .landing-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 6px;
    }
    .landing-subtitle {
        color: #6b7ab8;
        font-size: 0.88rem;
        text-align: center;
        margin-bottom: 36px;
    }
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 18px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2640;
    }
    </style>

    <div class="landing-container">
        <div class="landing-logo">
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="48" height="48" rx="12" fill="rgba(37,99,235,0.15)"/>
                <path d="M8 14C8 12.9 8.9 12 10 12H38C39.1 12 40 12.9 40 14V34C40 35.1 39.1 36 38 36H10C8.9 36 8 35.1 8 34V14Z" stroke="#3b82f6" stroke-width="1.8" fill="none"/>
                <path d="M8 14L24 26L40 14" stroke="#3b82f6" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="landing-title">Classification d'Emails</div>
        <div class="landing-subtitle">Système intelligent Français / Arabe &mdash; PFE 2025</div>
        <div class="section-label">Formulaire d'accès</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("**Nom complet / Nom de la société**")
        nom = st.text_input("", placeholder="Ex: Ahmed Benali / Groupe Oulmes", label_visibility="collapsed")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Adresse email**")
            email = st.text_input("", placeholder="contact@exemple.com", key="email_input", label_visibility="collapsed")
        with col2:
            st.markdown("**Téléphone**")
            tel = st.text_input("", placeholder="+212 6XX XXX XXX", key="tel_input", label_visibility="collapsed")

        st.markdown("**Secteur d'activité**")
        secteur = st.selectbox("", [
            "Sélectionner...",
            "Industrie / Production",
            "Commerce / Distribution",
            "Services / Conseil",
            "Administration / Public",
            "Santé / Médical",
            "Éducation / Formation",
            "Autre"
        ], label_visibility="collapsed")

        st.markdown("**Objet de l'utilisation**")
        objet = st.text_area(
            "",
            placeholder="Ex: Classification automatique des emails clients pour réduire le temps de traitement...",
            height=90,
            key="objet_input",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Accéder à l'application", use_container_width=True)

        if submitted:
            if not nom.strip():
                st.error("Veuillez entrer votre nom ou le nom de votre société.")
            elif not email.strip() or "@" not in email:
                st.error("Veuillez entrer une adresse email valide.")
            elif secteur == "Sélectionner...":
                st.error("Veuillez sélectionner votre secteur d'activité.")
            else:
                st.session_state.user_info = {
                    "nom": nom,
                    "email": email,
                    "tel": tel,
                    "secteur": secteur,
                    "objet": objet,
                    "date": datetime.datetime.now().strftime("%d/%m/%Y à %H:%M")
                }
                st.rerun()


if st.session_state.user_info is None:
    show_landing_page()
    st.stop()


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    u = st.session_state.user_info
    st.markdown(f"""
    <div style='background:#111827;border:1px solid #1e2640;border-radius:12px;padding:16px;margin-bottom:16px;'>
        <div style='font-family:Syne,sans-serif;font-size:0.75rem;color:#3b82f6;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>Session active</div>
        <div style='font-weight:600;color:#fff;font-size:0.95rem;'>{u['nom']}</div>
        <div style='color:#6b7ab8;font-size:0.8rem;margin-top:4px;'>{u['email']}</div>
        <div style='color:#6b7ab8;font-size:0.8rem;'>{u['secteur']}</div>
        <div style='color:#4b5563;font-size:0.75rem;margin-top:8px;'>Connecté le {u['date']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Deconnexion", use_container_width=True):
        st.session_state.user_info = None
        st.rerun()


# ══════════════════════════════════════════════════════════
# INITIALISATION MODELE
# ══════════════════════════════════════════════════════════
try:
    model = train_model()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du modèle : {e}")
    st.stop()


# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Analyse Unique",
    "Analyse Batch / CSV",
    "Analyse Image (OCR)",
    "Historique & Stats"
])


# ──────────────────────────────────────────────────────────
# TAB 1 : ANALYSE UNIQUE
# ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyse de texte direct")
    email_input = st.text_area("Collez l'email ici :", height=150, key="unique_input")

    if st.button("Analyser l'email"):
        if email_input.strip():
            result = classify(email_input, model)
            push_history(email_input, result)

            c1, c2, c3 = st.columns(3)
            c1.metric("Catégorie", result["pred"])
            c2.metric("Confiance", f"{result['confidence']:.0%}")
            c3.metric("Langue", result["langue"])

            fig = fig_scores(result["scores"])
            st.pyplot(fig)
            plt.close(fig)

            display_ner_section(email_input)
        else:
            st.warning("Veuillez saisir du texte.")


# ──────────────────────────────────────────────────────────
# TAB 2 : BATCH / CSV
# ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Classification de masse")
    source = st.radio("Sélectionnez la source :", ["Saisie Manuelle (lignes)", "Importer Fichier CSV"])
    lines_to_process = []

    if source == "Saisie Manuelle (lignes)":
        batch_txt = st.text_area("Entrez un email par ligne :", height=150)
        lines_to_process = [l.strip() for l in batch_txt.splitlines() if l.strip()]
    else:
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv", key="csv_upload")
        if uploaded_file:
            try:
                df_csv = pd.read_csv(uploaded_file)
                st.write("Aperçu :", df_csv.head(3))
                col_target = st.selectbox("Colonne contenant les emails :", df_csv.columns)
                lines_to_process = df_csv[col_target].dropna().astype(str).tolist()
            except Exception as e:
                st.error(f"Erreur de lecture du CSV : {e}")

    if st.button("Lancer l'analyse batch"):
        if lines_to_process:
            batch_results = []
            progress = st.progress(0)

            for i, line in enumerate(lines_to_process):
                res = classify(line, model)
                push_history(line, res)

                batch_results.append({
                    "Horodatage": datetime.datetime.now().strftime("%H:%M:%S"),
                    "Email": (str(line)[:50] + "...") if len(str(line)) > 50 else str(line),
                    "Categorie": res["pred"],
                    "Confiance (%)": round(res["confidence"] * 100, 1),
                    "Fiabilite": res["fiabilite"],
                    "Langue": res["langue"]
                })

                progress.progress((i + 1) / len(lines_to_process))

            st.success(f"Analyse terminée : {len(batch_results)} lignes traitées.")
            st.table(pd.DataFrame(batch_results))
        else:
            st.warning("Aucune ligne à traiter.")


# ──────────────────────────────────────────────────────────
# TAB 3 : IMAGE OCR
# ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Extraction de texte depuis Image")

    st.info(
        f"📎 Formats acceptés : **{', '.join(f'.{f}' for f in ACCEPTED_IMAGE_FORMATS)}**\n"
        "🔧 Preprocessing automatique : redressement, contraste, binarisation, débruitage"
    )

    img_file = st.file_uploader(
        "Charger une image (capture d'écran, photo, scan, document...)",
        type=ACCEPTED_IMAGE_FORMATS,
        key="img_upload"
    )

    img = None
    if img_file:
        try:
            img = Image.open(img_file)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
        except Exception as e:
            st.error(f"Impossible d'ouvrir l'image : {e}")
            img = None

    if img is not None:
        col_orig, col_proc = st.columns(2)

        with col_orig:
            st.markdown("**Image originale**")
            st.image(img, use_container_width=True)

        try:
            preview_processed = enhance_image_for_ocr(img)
            with col_proc:
                st.markdown("**Image après preprocessing**")
                st.image(preview_processed, use_container_width=True)
        except Exception:
            preview_processed = None

        with st.expander("⚙️ Options avancées OCR"):
            force_lang = st.selectbox(
                "Forcer la langue :",
                ["Auto (fra+ara)", "Français seulement", "Arabe seulement"]
            )
            show_details = st.checkbox("Afficher les détails du preprocessing", value=False)

        if st.button("Extraire et Classifier"):
            with st.spinner("Preprocessing et OCR en cours..."):
                try:
                    text_extracted, processed_img, mode_used = ocr_image(img)

                    if force_lang == "Français seulement":
                        text_extracted = pytesseract.image_to_string(
                            processed_img, lang="fra", config="--oem 3 --psm 3"
                        )
                        mode_used = "fra (forcé)"
                    elif force_lang == "Arabe seulement":
                        text_extracted = pytesseract.image_to_string(
                            processed_img, lang="ara", config="--oem 3 --psm 3"
                        )
                        mode_used = "ara (forcé)"

                    if show_details:
                        st.caption(f"Mode OCR utilisé : `{mode_used}` | Type détecté : `{detect_image_type(img)}`")

                    st.text_area("Texte détecté :", text_extracted, height=150)

                    if text_extracted.strip():
                        res_img = classify(text_extracted, model)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Catégorie", res_img["pred"])
                        c2.metric("Confiance", f"{res_img['confidence']:.0%}")
                        c3.metric("Langue détectée", res_img["langue"])

                        fig = fig_scores(res_img["scores"])
                        st.pyplot(fig)
                        plt.close(fig)

                        push_history("IMAGE_OCR: " + text_extracted, res_img)

                        if detect_language(text_extracted) == "Francais":
                            display_ner_section(text_extracted)
                    else:
                        st.warning(
                            "Aucun texte extrait. Conseils :\n"
                            "- Vérifiez que l'image est nette et bien éclairée\n"
                            "- Essayez une autre option de langue\n"
                            "- Assurez-vous que Tesseract est bien installé avec les packs ara et fra"
                        )

                except Exception as e:
                    st.error(
                        f"Erreur OCR : {e}\n\n"
                        "Vérifiez que Tesseract est installé : C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n"
                        "Et que les langues fra et ara sont installées."
                    )


# ──────────────────────────────────────────────────────────
# TAB 4 : HISTORIQUE
# ──────────────────────────────────────────────────────────
with tab4:
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        f_hist = fig_history(st.session_state.history)
        if f_hist:
            st.pyplot(f_hist)
            plt.close(f_hist)

        st.download_button(
            label="Exporter en CSV",
            data=build_csv(st.session_state.history),
            file_name="historique_pfe.csv",
            mime="text/csv"
        )

        if st.button("Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Aucune donnée dans l'historique.")
