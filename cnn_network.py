import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import process_image_for_cnn

def train_cnn_network(train_dir, test_dir, num_classes, initial_conv_filters, dense_neurons_str, epochs):
    """
    Treina uma rede neural convolucional para classificação de imagens
    
    Args:
        train_dir (str): Diretório contendo as imagens de treinamento (em subpastas por classe)
        test_dir (str): Diretório contendo as imagens de teste (em subpastas por classe)
        num_classes (int): Número de classes para classificação
        initial_conv_filters (int): Número de filtros iniciais na camada Conv2D
        dense_neurons_str (str): Neurônios nas camadas densas, formato: "128,64"
        epochs (int): Número de épocas para treinamento
    
    Returns:
        tuple: (modelo treinado, acurácia, histórico descartado, nomes das classes)
    """
    # Gerador para dados de treinamento com aumento
    gerador_treinamento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=7,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Carregar dados de treinamento
    base_treinamento = gerador_treinamento.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical'
    )
    
    # Gerador para dados de teste
    gerador_teste = ImageDataGenerator(rescale=1./255)
    
    # Carregar dados de teste
    base_teste = gerador_teste.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )
    
    # Construir modelo CNN
    rede_neural = Sequential()
    
    # Primeira camada convolucional + pooling
    rede_neural.add(Conv2D(
        initial_conv_filters, 
        (3, 3), 
        input_shape=(64, 64, 3), 
        activation='relu'
    ))
    rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Segunda camada convolucional + pooling
    rede_neural.add(Conv2D(
        initial_conv_filters, 
        (3, 3), 
        activation='relu'
    ))
    rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Achatamento para conectar às camadas densas
    rede_neural.add(Flatten())
    
    # Adicionar camadas densas conforme configuração
    dense_neurons_list = [int(n) for n in dense_neurons_str.split(',')]
    for neurons in dense_neurons_list:
        rede_neural.add(Dense(units=neurons, activation='relu'))
    
    # Camada de saída
    rede_neural.add(Dense(units=num_classes, activation='softmax'))
    
    # Compilar o modelo
    rede_neural.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar modelo
    rede_neural.fit(
        base_treinamento, 
        epochs=epochs, 
        validation_data=base_teste,
        verbose=1
    )
    
    # Avaliar o modelo
    loss, accuracy = rede_neural.evaluate(base_teste, verbose=0)
    
    # Mapear índices de classes para nomes
    class_indices = base_treinamento.class_indices
    class_names = list(class_indices.keys())
    
    return rede_neural, accuracy, None, class_names

def classify_image_cnn(model, image_path, class_names):
    """
    Classifica uma imagem usando o modelo CNN treinado
    
    Args:
        model: Modelo CNN treinado
        image_path (str): Caminho para a imagem a ser classificada
        class_names (list): Lista com os nomes das classes
    
    Returns:
        tuple: (classe_predita, probabilidade)
    """
    # Pré-processar a imagem
    img_array = process_image_for_cnn(image_path)
    if img_array is None:
        return None, 0
    
    # Fazer a previsão
    predictions = model.predict(img_array)[0]
    
    # Obter a classe com maior probabilidade
    pred_class_idx = np.argmax(predictions)
    pred_class = class_names[pred_class_idx]
    probability = float(predictions[pred_class_idx])
    
    return pred_class, probability 