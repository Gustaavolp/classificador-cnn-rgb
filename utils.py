import os
import tensorflow

def save_model(modelo, caminho):
    modelo.save(caminho)
    return True

def load_model(caminho):
    if not os.path.exists(caminho):
        return None
    return tensorflow.keras.models.load_model(caminho)

def ensure_dir(diretorio):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)