from tensorflow.keras import models
import tensorflow as tf
from inference_model import preprocess_image_for_model
import numpy as np
from PIL import Image

# Cargamos el modelo
model_final = tf.keras.models.load_model('CNN_final.keras')

# Cargamos la lista clases
clases = ["Conducción Segura",      # 0
        "Hablando por Teléfono",    # 1
        "Texteando por Teléfono",   # 2
        "Imprudencia al Volante",   # 3
        "Otro riesgo"               # 4
        ]

# path_img <- Obtenido de la web

img = preprocess_image_for_model(path_img)
prediction = model_final.predict(img)
predicted_class = np.argmax(prediction)

print("Clase predicha:", predicted_class, clases[predicted_class])