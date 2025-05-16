import os
import numpy
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def train_cnn_network(train_dir, test_dir, num_classes, initial_conv_filters, dense_neurons_str, epochs):

    # Gerador para dados de treinamento com aumento
    gerador_treinamento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2
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
        initial_conv_filters * 2, 
        (3, 3), 
        activation='relu'
    ))
    rede_neural.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Achatamento para conectar às camadas densas
    rede_neural.add(Flatten())
    
    # Adicionar camadas densas conforme configuração
    neurons = int(dense_neurons_str)  # converte para inteiro
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

def process_image_for_cnn(image_path, target_size=(64, 64)):
    """Pré-processa uma imagem para classificação com CNN"""
    try:
        if not image_path or not os.path.exists(image_path):
            print(f"Erro: Imagem não encontrada em {image_path}")
            return None
            
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = numpy.array(img) / 255.0
        img_array = numpy.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None

def classify_image_cnn(model, image_path, class_names):

    # Pré-processar a imagem
    img_array = process_image_for_cnn(image_path)
    if img_array is None:
        return None, 0
    
    # Fazer a previsão
    predictions = model.predict(img_array)[0]
    
    # Obter a classe com maior probabilidade
    pred_class_idx = numpy.argmax(predictions)
    pred_class = class_names[pred_class_idx]
    probability = float(predictions[pred_class_idx])
    
    return pred_class, probability 