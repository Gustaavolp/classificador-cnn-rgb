# Classificador CNN-RGB

Um aplicativo de classificação de imagens que utiliza duas abordagens diferentes de redes neurais: uma baseada em características RGB e outra usando uma Rede Neural Convolucional (CNN).

## Características

- **Classificação RGB**: Utiliza características de cor RGB para classificação
- **Classificação CNN**: Utiliza uma rede neural convolucional para classificação de imagens
- Interface gráfica intuitiva
- Suporte para treinamento, salvamento e carregamento de modelos
- Visualização de resultados em tempo real

## Requisitos

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Tkinter
- scikit-learn

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Gustaavolp/classificador-cnn-rgb.git
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

Execute o aplicativo principal:
```bash
python main_app.py
```

## Intervalos RGB para Classificação

### Homer Simpson
- Calça azul: R=30, G=100, B=180, Tolerância=40

### Bart Simpson
- Camisa laranja: R=255, G=120, B=0, Tolerância=40

## Estrutura do Projeto

- `main_app.py`: Aplicativo principal com interface gráfica
- `rgb_pixel_network.py`: Implementação da rede neural baseada em RGB
- `cnn_network.py`: Implementação da rede neural convolucional
- `utils.py`: Funções utilitárias

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes. 