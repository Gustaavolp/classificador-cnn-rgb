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
        
        for r_center, g_center, b_center, tolerance, _ in rgb_intervals:
            # Converter para inteiros
            r_center = int(r_center) if not isinstance(r_center, int) else r_center
            g_center = int(g_center) if not isinstance(g_center, int) else g_center
            b_center = int(b_center) if not isinstance(b_center, int) else b_center
            tolerance = int(tolerance) if not isinstance(tolerance, int) else tolerance
            
            # Criar máscaras para cada canal
            r_mask = (pixels[:,:,0] >= r_center - tolerance) & (pixels[:,:,0] <= r_center + tolerance)
            g_mask = (pixels[:,:,1] >= g_center - tolerance) & (pixels[:,:,1] <= g_center + tolerance)
            b_mask = (pixels[:,:,2] >= b_center - tolerance) & (pixels[:,:,2] <= b_center + tolerance)
            
            # Combinar máscaras para encontrar pixels dentro do intervalo RGB
            rgb_mask = r_mask & g_mask & b_mask
            
            # Calcular percentual de pixels neste intervalo
            matching_pixels = numpy.sum(rgb_mask)
            percentage = (matching_pixels / total_pixels) * 100
            features.append(percentage)
            
        return features
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None

def create_rgb_dataset(class_folders, rgb_intervals, output_csv):

    data = []
    
    for class_name, folder_path in class_folders.items():
        if not os.path.exists(folder_path):
            print(f"Pasta não encontrada: {folder_path}")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)
                features = extract_rgb_features(image_path, rgb_intervals)
                
                if features:
                    # Adicionar classe como último elemento
                    row = features + [class_name]
                    data.append(row)
    
    # Criar nomes das colunas
    feature_names = [str(interval[-1]) for interval in rgb_intervals]
    column_names = feature_names + ["classe"]

    # Criar DataFrame e salvar como CSV
    df = pandas.DataFrame(data, columns=column_names)
    df.to_csv(output_csv, index=False)
    
    return output_csv

def train_rgb_network(csv_path, num_hidden_layers, neurons_per_layer_str, epochs, test_split_ratio, num_classes):

    # Carregamento da base de dados
    dataset = pandas.read_csv(csv_path)

    # Separação dos dados em features (X) e target (y)
    num_features = len(dataset.columns) - 1
    X = dataset.iloc[:, 0:num_features].values
    y_original = dataset.iloc[:, num_features].values

    # Pré-processamento do target
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_original)
    class_names = encoder.classes_
    
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

    # Divisão dos dados em treino e teste
    X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
        X, y, test_size=test_split_ratio, stratify=y if num_classes > 1 else None)

    # Construção da rede neural
    rede_neural = tensorflow.keras.models.Sequential()
    neurons_list = [int(n) for n in neurons_per_layer_str.split(',')]
    
    # Primeira camada oculta
    rede_neural.add(tensorflow.keras.layers.Dense(
        units=neurons_list[0], 
        activation='relu', 
        input_shape=(num_features,)
    ))
    
    # Camadas ocultas adicionais
    for i in range(1, min(num_hidden_layers, len(neurons_list))):
        rede_neural.add(tensorflow.keras.layers.Dense(
            units=neurons_list[i], 
            activation='relu'
        ))
            
    # Camada de saída
    rede_neural.add(tensorflow.keras.layers.Dense(
        units=output_units, 
        activation=output_activation
    ))
    
    # Compilação da rede neural
    rede_neural.compile(
        optimizer='Adam', 
        loss=loss_function, 
        metrics=['accuracy']
    )
    
    # Treinamento da rede neural
    historico = rede_neural.fit(
        X_treinamento, 
        y_treinamento, 
        epochs=epochs, 
        validation_split=0.1, 
        verbose=1
    )
    
    # Avaliação do modelo
    loss, accuracy = rede_neural.evaluate(X_teste, y_teste, verbose=0)
    
    # Retornar modelo, acurácia e histórico
    return rede_neural, accuracy, historico, class_names

def classify_image_rgb(model, image_path, rgb_intervals, class_names):

    features = extract_rgb_features(image_path, rgb_intervals)
    if not features:
        return None, 0
        
    # Converter para array e preparar formato para previsão
    features_array = numpy.array([features])
    
    # Fazer previsão
    prediction = model.predict(features_array)[0]
    
    # Interpretar resultado (binário ou multiclasse)
    if len(prediction.shape) == 0 or prediction.shape[0] == 1:  # Modelo binário
        prob = float(prediction[0] if len(prediction.shape) > 0 else prediction)
        pred_class = 1 if prob >= 0.5 else 0
        return class_names[pred_class], prob if pred_class == 1 else 1 - prob
    else:  # Modelo multiclasse
        pred_class = numpy.argmax(prediction)
        return class_names[pred_class], prediction[pred_class] 