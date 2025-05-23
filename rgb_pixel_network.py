import os
import numpy
import pandas
import tensorflow
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def extract_rgb_features(image_path, rgb_intervals):
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = numpy.array(img)
        total_pixels = pixels.shape[0] * pixels.shape[1]
        features = []
        
        for r, g, b, tolerancia, _ in rgb_intervals:
            r = int(r) if not isinstance(r, int) else r
            g = int(g) if not isinstance(g, int) else g
            b = int(b) if not isinstance(b, int) else b
            tolerancia = int(tolerancia) if not isinstance(tolerancia, int) else tolerancia
            
            r_mask = (pixels[:,:,0] >= r - tolerancia) & (pixels[:,:,0] <= r + tolerancia)
            g_mask = (pixels[:,:,1] >= g - tolerancia) & (pixels[:,:,1] <= g + tolerancia)
            b_mask = (pixels[:,:,2] >= b - tolerancia) & (pixels[:,:,2] <= b + tolerancia)
            
            rgb_mask = r_mask & g_mask & b_mask
            
            pixels_matching = numpy.sum(rgb_mask)
            percent = (pixels_matching / total_pixels) * 100
            features.append(percent)
            
        return features
    except Exception as e:
        print(f"Erro na imagem {image_path}: {e}")
        return None

def create_rgb_dataset(class_folders, rgb_intervals, output_csv):
    data = []
    
    for classe, pasta in class_folders.items():
        if not os.path.exists(pasta):
            print(f"Pasta não existe: {pasta}")
            continue
            
        for arquivo in os.listdir(pasta):
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                caminho = os.path.join(pasta, arquivo)
                features = extract_rgb_features(caminho, rgb_intervals)
                
                if features:
                    row = features + [classe]
                    data.append(row)
    
    feature_names = [str(interval[-1]) for interval in rgb_intervals]
    colunas = feature_names + ["classe"]

    df = pandas.DataFrame(data, columns=colunas)
    df.to_csv(output_csv, index=False)
    
    return output_csv

def train_rgb_network(csv_path, num_hidden_layers, neurons_per_layer_str, epochs, test_split_ratio, num_classes):
    dataset = pandas.read_csv(csv_path)

    num_features = len(dataset.columns) - 1
    X = dataset.iloc[:, 0:num_features].values
    y_original = dataset.iloc[:, num_features].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_original)
    classes = encoder.classes_
    
    if num_classes == 2:
        y = y_encoded
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
        output_units = 1
    else:
        y = tensorflow.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
        output_activation = 'softmax'
        loss_function = 'categorical_crossentropy'
        output_units = num_classes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, stratify=y if num_classes > 1 else None)

    model = tensorflow.keras.models.Sequential()
    neurons = [int(n) for n in neurons_per_layer_str.split(',')]
    
    model.add(tensorflow.keras.layers.Dense(
        units=neurons[0], 
        activation='relu', 
        input_shape=(num_features,)
    ))
    
    for i in range(1, min(num_hidden_layers, len(neurons))):
        model.add(tensorflow.keras.layers.Dense(
            units=neurons[i], 
            activation='relu'
        ))
            
    model.add(tensorflow.keras.layers.Dense(
        units=output_units, 
        activation=output_activation
    ))
    
    model.compile(
        optimizer='Adam', 
        loss=loss_function, 
        metrics=['accuracy']
    )
    
    historico = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        validation_split=0.1, 
        verbose=1
    )
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return model, accuracy, historico, classes

def classify_image_rgb(model, image_path, rgb_intervals, class_names):
    features = extract_rgb_features(image_path, rgb_intervals)
    if not features:
        return None, 0
        
    features_array = numpy.array([features])
    
    prediction = model.predict(features_array)[0]
    
    if len(prediction.shape) == 0 or prediction.shape[0] == 1:
        prob = float(prediction[0] if len(prediction.shape) > 0 else prediction)
        classe = 1 if prob >= 0.5 else 0
        return class_names[classe], prob if classe == 1 else 1 - prob
    else:
        classe = numpy.argmax(prediction)
        return class_names[classe], prediction[classe] 