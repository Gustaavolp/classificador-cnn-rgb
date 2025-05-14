import os
import numpy as np
import tensorflow as tf
from PIL import Image

def save_model(model, model_path):
    """Salva o modelo treinado no caminho especificado"""
    model.save(model_path)
    return True

def load_model(model_path):
    """Carrega o modelo do caminho especificado"""
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def process_image_for_cnn(image_path, target_size=(64, 64)):
    """Pré-processa uma imagem para classificação com CNN"""
    try:
        if not image_path or not os.path.exists(image_path):
            print(f"Erro: Imagem não encontrada em {image_path}")
            return None
            
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None

def ensure_dir(directory):
    """Garante que um diretório existe"""
    if not os.path.exists(directory):
        os.makedirs(directory) 