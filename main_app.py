import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog
import tensorflow as tf
import tempfile
import shutil
from sklearn.model_selection import train_test_split

from rgb_pixel_network import extract_rgb_features, create_rgb_dataset, train_rgb_network, classify_image_rgb
from cnn_network import train_cnn_network, classify_image_cnn
from utils import save_model, load_model, ensure_dir

class ClassificadorImagensApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Imagens")
        self.root.geometry("900x700")
        
        # Variáveis de estado
        self.rgb_model = None
        self.rgb_class_names = None
        self.cnn_model = None
        self.cnn_class_names = None
        self.rgb_intervals = []
        self.class_folders = {}  # Para RGB
        self.cnn_class_folders = {}  # Para CNN
        self.current_image_path = None
        
        # Criar abas
        self.tab_control = ttk.Notebook(root)
        self.tab_rgb = ttk.Frame(self.tab_control)
        self.tab_cnn = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_rgb, text="RGB")
        self.tab_control.add(self.tab_cnn, text="CNN")
        self.tab_control.pack(expand=1, fill="both")
        
        # Área de log
        self.log_frame = ttk.LabelFrame(root, text="Log")
        self.log_frame.pack(fill="both", padx=5, pady=5)
        self.log_text = tk.Text(self.log_frame, height=5)
        self.log_text.pack(fill="both", expand=True)
        
        # Configurar abas
        self.setup_rgb_tab()
        self.setup_cnn_tab()
    
    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def setup_rgb_tab(self):
        # Frame para upload e processamento
        data_frame = ttk.LabelFrame(self.tab_rgb, text="Dados")
        data_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Botão para adicionar pasta
        ttk.Button(data_frame, text="Adicionar Pasta", 
                  command=lambda: self.add_class_folder("rgb")).grid(row=0, column=0, padx=5, pady=5)
        
        # Lista de classes
        ttk.Label(data_frame, text="Classes:").grid(row=0, column=1, padx=5)
        self.class_listbox = tk.Listbox(data_frame, width=40, height=5)
        self.class_listbox.grid(row=0, column=2, padx=5, pady=5, rowspan=3)
        
        # Frame para intervalos RGB
        rgb_frame = ttk.LabelFrame(data_frame, text="Intervalos RGB")
        rgb_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=5, sticky=tk.W+tk.E)
        
        # Entradas RGB
        ttk.Label(rgb_frame, text="Classe:").grid(row=0, column=0, padx=5)
        self.selected_class = ttk.Combobox(rgb_frame, width=15)
        self.selected_class.grid(row=0, column=1, padx=5)
        
        ttk.Label(rgb_frame, text="R:").grid(row=0, column=2, padx=5)
        self.r_value = ttk.Entry(rgb_frame, width=5)
        self.r_value.grid(row=0, column=3, padx=5)
        
        ttk.Label(rgb_frame, text="G:").grid(row=0, column=4, padx=5)
        self.g_value = ttk.Entry(rgb_frame, width=5)
        self.g_value.grid(row=0, column=5, padx=5)
        
        ttk.Label(rgb_frame, text="B:").grid(row=0, column=6, padx=5)
        self.b_value = ttk.Entry(rgb_frame, width=5)
        self.b_value.grid(row=0, column=7, padx=5)
        
        ttk.Label(rgb_frame, text="T:").grid(row=0, column=8, padx=5)
        self.tolerance_value = ttk.Entry(rgb_frame, width=5)
        self.tolerance_value.grid(row=0, column=9, padx=5)
        
        ttk.Button(rgb_frame, text="Adicionar", 
                  command=self.add_rgb_interval).grid(row=0, column=10, padx=5)
        
        # Lista de intervalos
        ttk.Label(rgb_frame, text="Intervalos:").grid(row=1, column=0, padx=5, columnspan=2)
        self.rgb_intervals_listbox = tk.Listbox(rgb_frame, width=60, height=5)
        self.rgb_intervals_listbox.grid(row=1, column=2, padx=5, pady=5, columnspan=9)
        
        # Botão processar
        ttk.Button(data_frame, text="Processar e Gerar CSV", 
                  command=self.process_images_and_generate_csv).grid(row=4, column=0, padx=5, pady=5, columnspan=2)
        
        # Frame para treinamento
        train_frame = ttk.LabelFrame(self.tab_rgb, text="Treinamento")
        train_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Parâmetros
        ttk.Label(train_frame, text="% teste:").grid(row=0, column=0, padx=5)
        self.test_split = ttk.Entry(train_frame, width=5)
        self.test_split.insert(0, "20")
        self.test_split.grid(row=0, column=1, padx=5)
        
        ttk.Label(train_frame, text="Camadas:").grid(row=0, column=2, padx=5)
        self.num_hidden_layers = ttk.Entry(train_frame, width=5)
        self.num_hidden_layers.insert(0, "2")
        self.num_hidden_layers.grid(row=0, column=3, padx=5)
        
        ttk.Label(train_frame, text="Neurônios:").grid(row=0, column=4, padx=5)
        self.neurons_per_layer = ttk.Entry(train_frame, width=10)
        self.neurons_per_layer.insert(0, "4,4")
        self.neurons_per_layer.grid(row=0, column=5, padx=5)
        
        ttk.Label(train_frame, text="Épocas:").grid(row=0, column=6, padx=5)
        self.epochs_rgb = ttk.Entry(train_frame, width=5)
        self.epochs_rgb.insert(0, "20")
        self.epochs_rgb.grid(row=0, column=7, padx=5)
        
        # Botões
        ttk.Button(train_frame, text="Treinar", command=self.train_rgb_model).grid(row=1, column=0, padx=5, pady=5, columnspan=2)
        ttk.Button(train_frame, text="Salvar", command=self.save_rgb_model).grid(row=1, column=2, padx=5, pady=5, columnspan=2)
        ttk.Button(train_frame, text="Carregar", command=self.load_rgb_model).grid(row=1, column=4, padx=5, pady=5, columnspan=2)
        
        # Frame para classificação
        classify_frame = ttk.LabelFrame(self.tab_rgb, text="Classificar")
        classify_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ttk.Button(classify_frame, text="Selecionar Imagem", 
                  command=lambda: self.select_image("rgb")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(classify_frame, text="Classificar", command=self.classify_rgb_image).grid(row=0, column=1, padx=5, pady=5)
        self.rgb_result_label = ttk.Label(classify_frame, text="Resultado: Nenhum")
        self.rgb_result_label.grid(row=0, column=2, padx=5, pady=5)

    def setup_cnn_tab(self):
        # Frame para dados
        data_frame = ttk.LabelFrame(self.tab_cnn, text="Dados")
        data_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Botão para adicionar pasta
        ttk.Button(data_frame, text="Adicionar Pasta", 
                  command=lambda: self.add_class_folder("cnn")).grid(row=0, column=0, padx=5, pady=5)
        
        # Lista de classes
        ttk.Label(data_frame, text="Classes:").grid(row=0, column=1, padx=5)
        self.cnn_class_listbox = tk.Listbox(data_frame, width=40, height=5)
        self.cnn_class_listbox.grid(row=0, column=2, padx=5, pady=5, rowspan=3)
        
        ttk.Label(data_frame, text="% teste:").grid(row=3, column=0, padx=5)
        self.cnn_test_split = ttk.Entry(data_frame, width=5)
        self.cnn_test_split.insert(0, "20")
        self.cnn_test_split.grid(row=3, column=1, padx=5)
        
        # Frame para parâmetros
        params_frame = ttk.LabelFrame(self.tab_cnn, text="Parâmetros")
        params_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ttk.Label(params_frame, text="Filtros:").grid(row=0, column=0, padx=5)
        self.initial_filters = ttk.Entry(params_frame, width=5)
        self.initial_filters.insert(0, "32")
        self.initial_filters.grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Neurônios:").grid(row=0, column=2, padx=5)
        self.dense_neurons = ttk.Entry(params_frame, width=10)
        self.dense_neurons.insert(0, "128,64")
        self.dense_neurons.grid(row=0, column=3, padx=5)
        
        ttk.Label(params_frame, text="Épocas:").grid(row=0, column=4, padx=5)
        self.epochs_cnn = ttk.Entry(params_frame, width=5)
        self.epochs_cnn.insert(0, "10")
        self.epochs_cnn.grid(row=0, column=5, padx=5)
        
        # Botões
        ttk.Button(params_frame, text="Treinar", command=self.train_cnn_model).grid(row=1, column=0, padx=5, pady=5, columnspan=2)
        ttk.Button(params_frame, text="Salvar", command=self.save_cnn_model).grid(row=1, column=2, padx=5, pady=5, columnspan=2)
        ttk.Button(params_frame, text="Carregar", command=self.load_cnn_model).grid(row=1, column=4, padx=5, pady=5, columnspan=2)
        
        # Frame para resultados
        self.results_frame = ttk.LabelFrame(self.tab_cnn, text="Resultados")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Frame para classificação
        classify_frame = ttk.LabelFrame(self.tab_cnn, text="Classificar")
        classify_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ttk.Button(classify_frame, text="Selecionar Imagem", 
                  command=lambda: self.select_image("cnn")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(classify_frame, text="Classificar", command=self.classify_cnn_image).grid(row=0, column=1, padx=5, pady=5)
        self.cnn_result_label = ttk.Label(classify_frame, text="Resultado: Nenhum")
        self.cnn_result_label.grid(row=0, column=2, padx=5, pady=5)
    
    def select_image(self, network_type):
        self.current_image_path = filedialog.askopenfilename(
            title="Selecionar Imagem", 
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if self.current_image_path:
            self.log_message(f"Imagem: {self.current_image_path}")
    
    def add_class_folder(self, network_type):
        folder_path = filedialog.askdirectory(title="Selecionar Pasta")
        if not folder_path:
            return
            
        class_name = os.path.basename(folder_path)
        
        if network_type == "rgb":
            self.class_folders[class_name] = folder_path
            self.class_listbox.insert(tk.END, f"{class_name}: {folder_path}")
            self.selected_class['values'] = list(self.class_folders.keys())
        else:  # CNN
            self.cnn_class_folders[class_name] = folder_path
            self.cnn_class_listbox.insert(tk.END, f"{class_name}: {folder_path}")
    
    def add_rgb_interval(self):
        class_name = self.selected_class.get()
        try:
            r = int(self.r_value.get())
            g = int(self.g_value.get())
            b = int(self.b_value.get())
            tolerance = int(self.tolerance_value.get())
            
            interval = [r, g, b, tolerance, class_name]
            self.rgb_intervals.append(interval)
            self.rgb_intervals_listbox.insert(tk.END, 
                f"{class_name}: R={r}, G={g}, B={b}, T={tolerance}")
            
            # Limpar campos
            self.r_value.delete(0, tk.END)
            self.g_value.delete(0, tk.END)
            self.b_value.delete(0, tk.END)
            self.tolerance_value.delete(0, tk.END)
        except ValueError:
            self.log_message("Erro: Os valores RGB e tolerância devem ser números inteiros")
    
    def process_images_and_generate_csv(self):
        output_csv = "rgb_features.csv"
        create_rgb_dataset(self.class_folders, self.rgb_intervals, output_csv)
        self.log_message(f"Dataset gerado: {output_csv}")
    
    def train_rgb_model(self):
        csv_path = "rgb_features.csv"
        try:
            test_split_ratio = float(self.test_split.get()) / 100
            num_hidden_layers = int(self.num_hidden_layers.get())
            neurons_per_layer_str = self.neurons_per_layer.get()
            epochs = int(self.epochs_rgb.get())
            
            df = pd.read_csv(csv_path)
            num_classes = len(df.iloc[:, -1].unique())
            
            self.log_message(f"Treinando com {num_classes} classes")
            self.rgb_model, accuracy, _, self.rgb_class_names = train_rgb_network(
                csv_path, num_hidden_layers, neurons_per_layer_str, 
                epochs, test_split_ratio, num_classes
            )
            self.log_message(f"Acurácia: {accuracy:.4f}")
        except Exception as e:
            self.log_message(f"Erro ao treinar: {str(e)}")
    
    def save_model(self, model_type):
        save_dir = filedialog.askdirectory(title="Salvar Modelo")
        if not save_dir:
            return
            
        ensure_dir(save_dir)
        
        try:
            if model_type == "rgb":
                model_path = os.path.join(save_dir, "rgb_model.h5")
                intervals_path = os.path.join(save_dir, "rgb_intervals.npy")
                classes_path = os.path.join(save_dir, "rgb_class_names.npy")
                
                save_model(self.rgb_model, model_path)
                np.save(intervals_path, self.rgb_intervals)
                np.save(classes_path, self.rgb_class_names)
            else:  # CNN
                model_path = os.path.join(save_dir, "cnn_model.h5")
                classes_path = os.path.join(save_dir, "cnn_class_names.npy")
                
                save_model(self.cnn_model, model_path)
                np.save(classes_path, self.cnn_class_names)
                
            self.log_message(f"Modelo {model_type.upper()} salvo em: {model_path}")
        except Exception as e:
            self.log_message(f"Erro ao salvar modelo: {str(e)}")
    
    def save_rgb_model(self):
        self.save_model("rgb")
    
    def save_cnn_model(self):
        self.save_model("cnn")
    
    def load_rgb_model(self):
        model_path = filedialog.askopenfilename(
            title="Carregar Modelo",
            filetypes=[("Modelo H5", "*.h5")]
        )
        if not model_path:
            return
            
        model_dir = os.path.dirname(model_path)
        intervals_path = os.path.join(model_dir, "rgb_intervals.npy")
        classes_path = os.path.join(model_dir, "rgb_class_names.npy")
        
        if not os.path.exists(intervals_path) or not os.path.exists(classes_path):
            self.log_message("Erro: Arquivos auxiliares não encontrados")
            return
            
        try:
            self.rgb_model = load_model(model_path)
            self.rgb_intervals = np.load(intervals_path, allow_pickle=True).tolist()
            self.rgb_class_names = np.load(classes_path, allow_pickle=True)
            
            self.rgb_intervals_listbox.delete(0, tk.END)
            for r, g, b, t, class_name in self.rgb_intervals:
                self.rgb_intervals_listbox.insert(tk.END, 
                    f"{class_name}: R={r}, G={g}, B={b}, T={t}")
            
            self.log_message(f"Modelo RGB carregado com {len(self.rgb_class_names)} classes")
        except Exception as e:
            self.log_message(f"Erro ao carregar modelo: {str(e)}")
    
    def load_cnn_model(self):
        model_path = filedialog.askopenfilename(
            title="Carregar Modelo",
            filetypes=[("Modelo H5", "*.h5")]
        )
        if not model_path:
            return
            
        model_dir = os.path.dirname(model_path)
        classes_path = os.path.join(model_dir, "cnn_class_names.npy")
        
        if not os.path.exists(classes_path):
            self.log_message("Erro: Arquivo de classes não encontrado")
            return
            
        try:
            self.cnn_model = load_model(model_path)
            self.cnn_class_names = np.load(classes_path, allow_pickle=True)
            self.log_message(f"Modelo CNN carregado com {len(self.cnn_class_names)} classes")
        except Exception as e:
            self.log_message(f"Erro ao carregar modelo: {str(e)}")
    
    def train_cnn_model(self):
        try:
            if not self.cnn_class_folders:
                self.log_message("Erro: Adicione pastas de classes primeiro")
                return
                
            num_classes = len(self.cnn_class_folders)
            initial_filters = int(self.initial_filters.get())
            dense_neurons_str = self.dense_neurons.get()
            epochs = int(self.epochs_cnn.get())
            test_split = float(self.cnn_test_split.get()) / 100
            
            # Limpar frame de resultados
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            
            # Criar diretórios temporários
            temp_dir = tempfile.mkdtemp()
            train_dir = os.path.join(temp_dir, 'train')
            test_dir = os.path.join(temp_dir, 'test')
            os.makedirs(train_dir)
            os.makedirs(test_dir)
            
            # Processar cada classe
            for class_name, folder_path in self.cnn_class_folders.items():
                # Criar diretórios
                train_class_dir = os.path.join(train_dir, class_name)
                test_class_dir = os.path.join(test_dir, class_name)
                os.makedirs(train_class_dir)
                os.makedirs(test_class_dir)
                
                # Listar imagens
                images = [f for f in os.listdir(folder_path) 
                         if os.path.isfile(os.path.join(folder_path, f)) and
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                # Dividir imagens
                train_images, test_images = train_test_split(images, test_size=test_split, random_state=42)
                
                # Copiar imagens
                for img in train_images:
                    shutil.copy2(os.path.join(folder_path, img), os.path.join(train_class_dir, img))
                
                for img in test_images:
                    shutil.copy2(os.path.join(folder_path, img), os.path.join(test_class_dir, img))
                
                self.log_message(f"{class_name}: {len(train_images)} treino, {len(test_images)} teste")
            
            # Treinar modelo
            self.cnn_model, accuracy, _, self.cnn_class_names = train_cnn_network(
                train_dir, test_dir, num_classes, 
                initial_filters, dense_neurons_str, epochs
            )
            
            # Limpar diretórios
            shutil.rmtree(temp_dir)
            
            # Mostrar resultados
            self.display_cnn_results(accuracy)
            
        except Exception as e:
            self.log_message(f"Erro ao treinar CNN: {str(e)}")
    
    def display_cnn_results(self, accuracy):
        # Mostrar acurácia
        accuracy_label = ttk.Label(self.results_frame, text=f"Acurácia: {accuracy:.4f}")
        accuracy_label.grid(row=0, column=0, padx=5, pady=5)
        self.log_message(f"Treinamento concluído: {accuracy:.4f}")
    
    def classify_rgb_image(self):
        if not self.current_image_path:
            self.log_message("Erro: Selecione uma imagem primeiro")
            return
        
        if not self.rgb_model:
            self.log_message("Erro: Treine ou carregue um modelo primeiro")
            return
            
        class_name, probability = classify_image_rgb(
            self.rgb_model, self.current_image_path, 
            self.rgb_intervals, self.rgb_class_names
        )
        
        if class_name is None:
            self.rgb_result_label.config(text="Resultado: Erro ao processar imagem")
            self.log_message("Erro ao processar a imagem para classificação RGB")
        else:
            self.rgb_result_label.config(text=f"Resultado: {class_name} ({probability:.2%})")
            self.log_message(f"RGB: {class_name} ({probability:.2%})")
    
    def classify_cnn_image(self):
        if not self.current_image_path:
            self.log_message("Erro: Selecione uma imagem primeiro")
            return
            
        if not self.cnn_model:
            self.log_message("Erro: Treine ou carregue um modelo primeiro")
            return
            
        class_name, probability = classify_image_cnn(
            self.cnn_model, self.current_image_path, self.cnn_class_names
        )
        
        if class_name is None:
            self.cnn_result_label.config(text="Resultado: Erro ao processar imagem")
            self.log_message("Erro ao processar a imagem para classificação CNN")
        else:
            self.cnn_result_label.config(text=f"Resultado: {class_name} ({probability:.2%})")
            self.log_message(f"CNN: {class_name} ({probability:.2%})")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassificadorImagensApp(root)
    root.mainloop() 