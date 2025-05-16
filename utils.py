import os
import tensorflow

def save_model(model, model_path):
    model.save(model_path)
    return True

def load_model(model_path):
    if os.path.exists(model_path):
        return tensorflow.keras.models.load_model(model_path)
    return None

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)