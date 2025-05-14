# classificador-cnn-rgb

Sistema de classificação de imagens com duas abordagens:

1. **Rede RGB**: Classifica imagens com base em características de cores RGB
2. **Rede CNN**: Utiliza redes neurais convolucionais para classificação direta

## Valores RGB recomendados

### Para Homer Simpson:
- Calça azul: R=30, G=100, B=180, Tolerância=40

### Para Bart Simpson:
- Camisa laranja: R=255, G=120, B=0, Tolerância=40

## Funcionalidades

- Interface gráfica com abas separadas para cada tipo de rede
- Treinamento, salvamento e carregamento de modelos
- Visualização de resultados de classificação
- Possibilidade de adicionar novas classes

## Requisitos

- Python 3.x
- Tensorflow
- Numpy
- Pandas
- Pillow
- scikit-learn

Instale as dependências com:

```bash
pip install tensorflow numpy pandas pillow matplotlib scikit-learn
```

## Estrutura do Projeto

- `main_app.py`: Interface gráfica principal da aplicação
- `rgb_pixel_network.py`: Implementação da rede neural baseada em pixels RGB
- `cnn_network.py`: Implementação da rede neural convolucional
- `utils.py`: Funções utilitárias compartilhadas

## Como Executar

```bash
python main_app.py
```

## Como Usar

### Rede RGB

1. Clique em "Adicionar Pasta de Classe" para cada classe de imagens
2. Defina intervalos RGB para cada classe usando os campos R, G, B e Tolerância
3. Clique em "Processar Imagens e Gerar CSV"
4. Configure os parâmetros da rede e clique em "Treinar Rede RGB"
5. Após o treinamento, salve o modelo ou use-o para classificar novas imagens

### Rede CNN

1. Clique em "Selecionar Pasta de Treino" e "Selecionar Pasta de Teste"
2. Configure os parâmetros da rede e clique em "Treinar Rede CNN"
3. Após o treinamento, salve o modelo ou use-o para classificar novas imagens

## Notas Importantes

- Estrutura de diretórios para CNN: as imagens devem estar organizadas em subpastas nomeadas de acordo com as classes
- Para o modelo RGB, recomenda-se definir pelo menos um intervalo RGB para cada classe
- Os modelos treinados são salvos na pasta "models" 