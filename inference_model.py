import numpy as np
from PIL import Image

# Función para preprocesar una imagen en producción
def preprocess_image_for_model(file_path):
    """
    Preprocesa una imagen para que sea compatible con el modelo de TensorFlow.
    Args:
        file_path (str): Ruta al archivo de imagen (en la web que se sube la imagen).
    Returns:
        np.array: Imagen preprocesada como un array de numpy.
    """
    # 'L' para grayscale
    img = Image.open(file_path).convert("L")  
    img = img.resize((200, 200))
    img_array = np.array(img).astype("float32") / 255.0  # normalizar
    img_array = np.expand_dims(img_array, axis=-1)  # (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 1)
    return img_array