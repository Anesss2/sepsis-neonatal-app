import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import csv
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sepsis Néonatal — Prédiction",
    page_icon="🩺",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH     = "saved_models/GradientBoosting_model.pkl"
THRESHOLD      = 0.15   # Best_Threshold from training

# ── Decision zones ────────────────────────────────────────────────────────────
# < ZONE_LOW            → NO SEPSIS
# ZONE_LOW to ZONE_HIGH → UNCERTAIN → "consulter un spécialiste"
# > ZONE_HIGH           → SEPSIS
ZONE_LOW       = 0.10   # below this = clear NO SEPSIS
ZONE_HIGH      = 0.20   # above this = clear SEPSIS alert

LOG_FILE       = "predictions_log.csv"

# ── Exact features from X.columns.tolist() ────────────────────────────────────
FEATURES = [
    'AGE_JOURS_ADMISSION',
    'ICD_PREMATURITE',
    'ICD_DETRESSE_RESPIRATOIRE',
    'ICD_ICTERE',
    'TEMPERATURE_MIN',
    'TEMPERATURE_MAX',
    'FC_MOYENNE',
    'GB_MIN',
    'PLAQUETTES_MAX',
    'BILIRUBINE_TOTALE_MIN',
    'CRP',
    'FR_MIN',
    'FR_MAX',
    'GLYCEMIE_MIN',
    'GLYCEMIE_MAX',
    'POIDS_NAISSANCE',
    'SEXE_M',   # one-hot encoded: 1=Masculin, 0=Féminin
]

# ══════════════════════════════════════════════════════════════════════════════
# LABELS & NORMAL RANGES
# ══════════════════════════════════════════════════════════════════════════════
LABELS = {
    'AGE_JOURS_ADMISSION'      : "Âge à l'admission (jours)",
    'SEXE_M'                   : "Sexe",
    'ICD_PREMATURITE'          : "Prématurité diagnostiquée (ICD-9)",
    'ICD_DETRESSE_RESPIRATOIRE': "Détresse respiratoire (ICD-9)",
    'ICD_ICTERE'               : "Ictère néonatal (ICD-9)",
    'TEMPERATURE_MIN'          : "Température minimale (°C)",
    'TEMPERATURE_MAX'          : "Température maximale (°C)",
    'FC_MOYENNE'               : "Fréquence cardiaque moyenne (bpm)",
    'GB_MIN'                   : "Globules blancs minimaux — leucopénie (K/μL)",
    'PLAQUETTES_MAX'           : "Plaquettes maximales — thrombocytose (K/μL)",
    'BILIRUBINE_TOTALE_MIN'    : "Bilirubine totale minimale (mg/dL)",
    'CRP'                      : "CRP — Protéine C-réactive (mg/L)",
    'FR_MIN'                   : "Fréquence respiratoire minimale (resp/min)",
    'FR_MAX'                   : "Fréquence respiratoire maximale (resp/min)",
    'GLYCEMIE_MIN'             : "Glycémie minimale — hypoglycémie (mg/dL)",
    'GLYCEMIE_MAX'             : "Glycémie maximale — hyperglycémie (mg/dL)",
    'POIDS_NAISSANCE'          : "Poids de naissance (kg)",
}

NORMAL_RANGES = {
    'AGE_JOURS_ADMISSION'      : (0,    3,     "jours",      "Nouveau-né : 0–3 jours"),
    'TEMPERATURE_MIN'          : (36.5, 37.5,  "°C",         "Normal : 36.5–37.5°C"),
    'TEMPERATURE_MAX'          : (36.5, 37.5,  "°C",         "Normal : 36.5–37.5°C"),
    'FC_MOYENNE'               : (120,  160,   "bpm",        "Normal nouveau-né : 120–160 bpm"),
    'GB_MIN'                   : (9.0,  30.0,  "K/μL",       "Normal : 9–30 K/μL · Leucopénie si < 5"),
    'PLAQUETTES_MAX'           : (150,  400,   "K/μL",       "Normal : 150–400 K/μL"),
    'BILIRUBINE_TOTALE_MIN'    : (0,    12,    "mg/dL",      "Ictère si > 12 mg/dL"),
    'CRP'                      : (0,    5,     "mg/L",       "Normal : < 5 · Infection si > 20"),
    'FR_MIN'                   : (30,   60,    "resp/min",   "Normal : 30–60 · Tachypnée si > 60"),
    'FR_MAX'                   : (30,   60,    "resp/min",   "Normal : 30–60 · Tachypnée si > 60"),
    'GLYCEMIE_MIN'             : (47,   100,   "mg/dL",      "Hypoglycémie si < 47 mg/dL"),
    'GLYCEMIE_MAX'             : (70,   150,   "mg/dL",      "Hyperglycémie si > 150 mg/dL"),
    'POIDS_NAISSANCE'          : (1.0,  4.5,   "kg",         "Faible poids si < 2.5 kg · Très faible si < 1.5 kg"),
}

BINARY_COLS = ['ICD_PREMATURITE', 'ICD_DETRESSE_RESPIRATOIRE', 'ICD_ICTERE']

LOG_FILE = "predictions_log.csv"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL — cached
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modèle introuvable : `{MODEL_PATH}`")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# LOG
# ══════════════════════════════════════════════════════════════════════════════
def save_log(patient_dict, prob, verdict):
    file_exists = os.path.exists(LOG_FILE)
    fields = ["timestamp"] + list(patient_dict.keys()) + ["probabilite_%", "verdict"]
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        row.update({k: ("" if pd.isna(v) else v) for k, v in patient_dict.items()})
        row["probabilite_%"] = round(prob * 100, 2)
        row["verdict"]       = verdict
        writer.writerow(row)



# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🩺 Prédiction du Sepsis Néonatal")
st.markdown(
    "Remplissez les données disponibles pour ce patient. "
    "Les champs laissés vides sont traités comme **inconnus**."
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
patient_dict = {}   # stores raw user values (before one-hot)

# ── Démographie ───────────────────────────────────────────────────────────────
st.subheader("🧬 Démographie")
c1, c2, c3 = st.columns(3)

with c1:
    sexe_val = st.selectbox("Sexe", ["— inconnu —", "Masculin (M)", "Féminin (F)"])
    patient_dict['SEXE'] = (
        np.nan if sexe_val == "— inconnu —" else
        "M" if "Masculin" in sexe_val else "F"
    )

with c2:
    v = st.text_input(
        LABELS['AGE_JOURS_ADMISSION'],
        placeholder=NORMAL_RANGES['AGE_JOURS_ADMISSION'][3]
    )
    patient_dict['AGE_JOURS_ADMISSION'] = float(v.replace(",",".")) if v.strip() else np.nan

with c3:
    v = st.text_input(
        LABELS['POIDS_NAISSANCE'],
        placeholder=NORMAL_RANGES['POIDS_NAISSANCE'][3]
    )
    patient_dict['POIDS_NAISSANCE'] = float(v.replace(",",".")) if v.strip() else np.nan

st.markdown("---")

# ── ICD Flags ─────────────────────────────────────────────────────────────────
st.subheader("🏥 Diagnostics ICD-9")
c1, c2, c3 = st.columns(3)

for col, ui_col in zip(BINARY_COLS, [c1, c2, c3]):
    with ui_col:
        v = st.selectbox(LABELS[col], ["— inconnu —", "Non (0)", "Oui (1)"], key=col)
        patient_dict[col] = (
            np.nan if v == "— inconnu —" else
            1 if v == "Oui (1)" else 0
        )

st.markdown("---")

# ── Constantes vitales ────────────────────────────────────────────────────────
st.subheader("🌡️ Constantes vitales")
c1, c2, c3 = st.columns(3)

vitals = [
    ('TEMPERATURE_MIN', c1),
    ('TEMPERATURE_MAX', c2),
    ('FC_MOYENNE',      c3),
    ('FR_MIN',          c1),
    ('FR_MAX',          c2),
]
for col, ui_col in vitals:
    with ui_col:
        v = st.text_input(LABELS[col], placeholder=NORMAL_RANGES[col][3], key=col)
        patient_dict[col] = float(v.replace(",",".")) if v.strip() else np.nan

st.markdown("---")

# ── Biologie ──────────────────────────────────────────────────────────────────
st.subheader("🔬 Biologie")
c1, c2, c3 = st.columns(3)

bio = [
    ('GB_MIN',              c1),
    ('PLAQUETTES_MAX',      c2),
    ('BILIRUBINE_TOTALE_MIN', c3),
    ('GLYCEMIE_MIN',        c1),
    ('GLYCEMIE_MAX',        c2),
    ('CRP',                 c3),
]
for col, ui_col in bio:
    with ui_col:
        nr = NORMAL_RANGES.get(col)
        placeholder = nr[3] if nr else "Vide = inconnu"
        v = st.text_input(LABELS[col], placeholder=placeholder, key=col)
        patient_dict[col] = float(v.replace(",",".")) if v.strip() else np.nan

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD INPUT FOR MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_model_input(patient_dict):
    """
    Build a DataFrame with exactly the columns from X.columns (FEATURES list).
    Handles one-hot encoding of SEXE → SEXE_M.
    """
    inp = pd.DataFrame(np.nan, index=[0], columns=FEATURES)

    for col, val in patient_dict.items():
        if pd.isna(val):
            continue

        if col == 'SEXE':
            # One-hot: SEXE_M = 1 if Male, 0 if Female
            if 'SEXE_M' in FEATURES:
                inp.loc[0, 'SEXE_M'] = 1 if val == "M" else 0
        elif col in FEATURES:
            inp.loc[0, col] = val

    return inp


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
enable_log = st.checkbox("💾 Sauvegarder les prédictions", value=True)

predict_clicked = st.button(
    "🔍 Lancer la prédiction",
    type="primary",
    use_container_width=True,
)

if predict_clicked:

    # ── Missing check ─────────────────────────────────────────────────────────
    n_miss  = sum(pd.isna(v) for v in patient_dict.values())
    n_total = len(patient_dict)
    pct     = n_miss / n_total * 100

    if pct > 70:
        st.error(f"❌ **{n_miss}/{n_total} champs manquants ({pct:.0f}%)** — résultats peu fiables.")
    elif pct > 40:
        st.warning(f"⚠️ **{n_miss}/{n_total} champs manquants ({pct:.0f}%)** — prédiction moins fiable.")
    else:
        st.success(f"✅ **{n_total - n_miss}/{n_total} champs renseignés** ({100-pct:.0f}%)")

    # ── Run model ─────────────────────────────────────────────────────────────
    inp  = build_model_input(patient_dict)
    prob = float(model.predict_proba(inp)[0][1])
    prob_pct = prob * 100

    st.markdown("## 📊 Résultat")

    # ── Progress bar ──────────────────────────────────────────────────────────
    st.markdown(f"### Probabilité de sepsis : **{prob_pct:.1f}%**")
    st.progress(min(prob, 1.0))

    col_info = st.columns(3)
    col_info[0].metric("Probabilité",  f"{prob_pct:.1f}%")
    col_info[1].metric("Seuil modèle", f"{THRESHOLD*100:.0f}%")
    col_info[2].metric("Zone incertaine", f"{ZONE_LOW*100:.0f}–{ZONE_HIGH*100:.0f}%")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # VERDICT — 3 zones
    # ══════════════════════════════════════════════════════════════════════════

    if prob < ZONE_LOW:
        # ── Zone 1: NO SEPSIS ─────────────────────────────────────────────────
        verdict = "PAS DE SEPSIS"
        st.success(
            f"## 🟢 PAS DE SEPSIS DÉTECTÉ\n\n"
            f"Probabilité : **{prob_pct:.1f}%** — en dessous du seuil de {ZONE_LOW*100:.0f}%\n\n"
            f"→ Aucun signe prédictif de sepsis selon le modèle.\n\n"
            f"→ Continuer la surveillance clinique de routine."
        )

    elif ZONE_LOW <= prob <= ZONE_HIGH:
        # ── Zone 2: UNCERTAIN ────────────────────────────────────────────────
        verdict = "ZONE INCERTAINE"
        st.warning(
            f"## 🟡 ZONE INCERTAINE\n\n"
            f"Probabilité : **{prob_pct:.1f}%** — entre {ZONE_LOW*100:.0f}% et {ZONE_HIGH*100:.0f}%\n\n"
            f"⚠️ **Il est préférable de consulter un spécialiste en néonatologie.**\n\n"
            f"→ Le modèle détecte des signaux faibles mais insuffisants pour conclure.\n\n"
            f"→ Évaluation clinique approfondie recommandée.\n\n"
            f"→ Envisager : hémoculture, NFS complète, CRP, bilan infectieux."
        )

    else:
        # ── Zone 3: SEPSIS ────────────────────────────────────────────────────
        verdict = "SEPSIS PROBABLE"
        st.error(
            f"## 🔴 SEPSIS PROBABLE\n\n"
            f"Probabilité : **{prob_pct:.1f}%** — au-dessus du seuil de {ZONE_HIGH*100:.0f}%\n\n"
            f"🚨 **Prise en charge médicale immédiate recommandée.**\n\n"
            f"→ Hémoculture en urgence avant toute antibiothérapie.\n\n"
            f"→ Bilan biologique complet : NFS, CRP, procalcitonine, bilan métabolique.\n\n"
            f"→ Envisager antibiothérapie empirique selon protocole NICU."
        )

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.info(
        "⚕️ **Avertissement médical** — Ce résultat est une aide à la décision "
        "basée sur un modèle de machine learning"
        "Il ne remplace pas le jugement clinique d'un médecin spécialiste."
    )

    # ── Log ───────────────────────────────────────────────────────────────────
    if enable_log:
        save_log(patient_dict, prob, verdict)

    # ── Data summary ──────────────────────────────────────────────────────────
    with st.expander("🔎 Données saisies pour ce patient"):
        summary = {}
        for k, v in patient_dict.items():
            label = LABELS.get(k, k)
            summary[label] = "— inconnu —" if pd.isna(v) else v
        st.dataframe(
            pd.DataFrame(summary, index=["Valeur"]).T,
            use_container_width=True,
        )
