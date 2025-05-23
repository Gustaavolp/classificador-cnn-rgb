import os
import numpy
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def train_cnn_network(train_dir, test_dir, num_classes, num_conv_layers, dense_neurons_str, epochs):
    # Gerador com aumento de dados
    gerador_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Carregar imagens
    base_train = gerador_train.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical'
    )
    
    base_teste = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical',
        shuffle=False
    )   

    # Modelo CNN básico
    modelo = Sequential()
    
    # Primeira camada (sempre presente)
    modelo.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Adicionar camadas convolucionais adicionais conforme solicitado
    for i in range(1, num_conv_layers):
        modelo.add(Conv2D(32, (3, 3), activation='relu'))
        modelo.add(MaxPooling2D(pool_size=(2, 2)))        

    modelo.add(Flatten())
    
    # Camada densa
    neurons = int(dense_neurons_str)
    modelo.add(Dense(units=neurons, activation='relu'))
    
    # Camada de saída
    modelo.add(Dense(units=num_classes, activation='softmax'))
    
    # Compilação
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Treino
    modelo.fit(base_train, epochs=epochs, validation_data=base_teste, verbose=1)
    
    loss, accuracy = modelo.evaluate(base_teste, verbose=0)
    
    class_names = list(base_train.class_indices.keys())
    
    return modelo, accuracy, None, class_names

def process_image_for_cnn(image_path, target_size=(64, 64)):
    try:
        if not os.path.exists(image_path):
            print(f"Arquivo não encontrado: {image_path}")
            return None
            
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = numpy.array(img) / 255.0
        img_array = numpy.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Erro: {e}")
        return None

def classify_image_cnn(model, image_path, class_names):
    img_array = process_image_for_cnn(image_path)
    if img_array is None:
        return None, 0
    
    predictions = model.predict(img_array)[0]
    
    indice = numpy.argmax(predictions)
    classe = class_names[indice]
    prob = float(predictions[indice])
    
    return classe, prob 