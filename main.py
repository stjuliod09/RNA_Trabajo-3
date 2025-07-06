from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn

# Función de pre-proceso para el clasificador de imágenes
from inference_model import preprocess_image_for_model
from tensorflow.keras.models import load_model

# -------------------------------------------------------------------
# Configuración general
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 1.  Clasificador de conducción distraída (TensorFlow)
# -------------------------------------------------------------------
CLASSES = [
    "Conducción Segura",
    "Hablando por Teléfono",
    "Texteando por Teléfono",
    "Imprudencia al Volante",
    "Otro riesgo",
]
classification_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "CNN_final.keras")
)

# -------------------------------------------------------------------
# 2.  Predictor de demanda (PyTorch)
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))


class PrediccionDemanda(nn.Module):
    """Red neuronal: 23 → 5 → 10 → 20 → 1  (ajusta si cambian los pesos)."""

    def __init__(self, n_input_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input_features, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Cargamos los pesos (state_dict)
state_dict = torch.load(
    os.path.join(MODELS_DIR, "modelo_dem_pytorch.pth"), map_location=device
)

# Averiguamos cuántas columnas produce el preprocesador
n_features = len(preprocessor.get_feature_names_out())

demand_model = PrediccionDemanda(n_features).to(device)
demand_model.load_state_dict(state_dict)
demand_model.eval()  # modo inferencia


# -------------------------------------------------------------------
# Función de inferencia para demanda
# -------------------------------------------------------------------
def predict_demand(raw_form: dict) -> float:
    """Convierte el formulario en DataFrame → tensor → predicción escalar."""
    df = pd.DataFrame([raw_form])  # una sola fila
    X = preprocessor.transform(df)  # ndarray (1, n_features)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = demand_model(X_tensor).cpu().numpy().squeeze()

    # Si el target estaba en alguna escala diferente, des-normalízalo aquí
    return float(y_pred)


# -------------------------------------------------------------------
# Función de recomendacion
# -------------------------------------------------------------------
all_dest_df = pd.read_csv(os.path.join(MODELS_DIR, "Expanded_Destinations.csv"))

dest_pop = all_dest_df[["Popularity"]].to_numpy(dtype=float)  # (N_dest,1)

dest_type_dummies = pd.get_dummies(
    all_dest_df["Type"],
    prefix="Type"
).astype(float)

dest_time_dummies = pd.get_dummies(
    all_dest_df["BestTimeToVisit"],
    prefix="BestTime"
).astype(float)

dest_features_array = np.hstack([
    dest_type_dummies.to_numpy(dtype=float),
    dest_time_dummies.to_numpy(dtype=float)
])  

N_dest = dest_features_array.shape[0]

reco_model  = load_model(os.path.join(MODELS_DIR,
                                      "travel_recommendation_model.keras"))
reco_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
reco_mlb    = joblib.load(os.path.join(MODELS_DIR, "mlb.joblib"))
gender_enc  = joblib.load(os.path.join(MODELS_DIR, "gender_encoder.joblib"))

def recommend_destinations(user, top_k=5):
    user_vals = np.array([
        user["NumberOfAdults"],
        user["NumberOfChildren"]
    ], dtype=float).reshape(1, -1)                     # (1,2)

    num_tab = np.hstack([
        np.repeat(user_vals, N_dest, axis=0),         # (N_dest,2)
        dest_pop                                      # (N_dest,1)
    ])                                                # → (N_dest,3)

    num_scaled = reco_scaler.transform(num_tab)       # (N_dest,3)

    g_arr = gender_enc.transform([user["Gender"]])    # (1,)
    g_rep = np.repeat(g_arr.reshape(-1,1), N_dest, axis=0)  # (N_dest,1)

    prefs = [p.strip() for p in user["Preferences"].split(",")]
    prefs_hot = reco_mlb.transform([prefs])           # (1,M)
    prefs_rep = np.repeat(prefs_hot, N_dest, axis=0)   # (N_dest,M)

    X_tab = np.hstack([num_scaled, g_rep, prefs_rep, dest_features_array])

    X_dest = all_dest_df[["DestinationID"]].to_numpy(dtype=int)

    scores = reco_model.predict([X_dest, X_tab]).flatten()

    df_scores = pd.DataFrame({
        "Name":  all_dest_df["Name"],
        "State": all_dest_df["State"],
        "Score": scores
    })

    top_df = (
        df_scores
        .sort_values("Score", ascending=False)
        .drop_duplicates(subset=["Name", "State"])  
        .head(top_k)
        .reset_index(drop=True)
    )

    return top_df[["Name", "State"]]



# -------------------------------------------------------------------
# Rutas Flask
# -------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------- Clasificador de imágenes ----------
@app.route("/classify", methods=["GET", "POST"])
def classify():
    label = filename = error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            error = "No se seleccionó imagen."
        else:
            original_name = secure_filename(file.filename)
            base, _ = os.path.splitext(original_name)
            filename = f"{base}.png"
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            try:
                img = Image.open(path).convert("RGB")
                img.save(path, format="PNG")  # normaliza extensión
                img_arr = preprocess_image_for_model(path)
                preds = classification_model.predict(img_arr)
                label = CLASSES[np.argmax(preds)]
            except (UnidentifiedImageError, Exception):
                os.remove(path)
                filename = None
                error = "Error al procesar la imagen."

    return render_template("classify.html", label=label, filename=filename, error=error)


# ---------- Predicción de demanda ----------
@app.route("/demand", methods=["GET", "POST"])
def demand():
    prediction = error = None

    if request.method == "POST":
        try:
            numeric_fields = {"max_capacity", "day_of_week", "month", "hour"}
            form_data = {k: v for k, v in request.form.items()}
            for f in numeric_fields:
                form_data[f] = int(form_data[f])  # o float() si tu modelo usa float

            prediction = predict_demand(form_data)

        except Exception as err:
            error = f"No se pudo procesar la solicitud: {err}"

    return render_template("demand.html", prediction=prediction, error=error)

@app.route("/recommend", methods=["GET","POST"])
def recommend():
    result = error = None
    if request.method == "POST":
        try:
            user = {
            'Preferences':      request.form['prefs'],
            'Gender':           request.form['gender'],
            'NumberOfAdults':   int(request.form['adults']),
            'NumberOfChildren': int(request.form['kids']),
            }
            df_rec = recommend_destinations(user, top_k=5)
            result = df_rec.to_dict(orient='records')
        except Exception as e:
            error = f"Hubo un problema: {e}"
    return render_template("recommend.html", result=result, error=error)
   
# -------------------------------------------------------------------
# Arranque local
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
