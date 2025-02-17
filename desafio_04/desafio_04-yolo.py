from ultralytics import YOLO

# Carregar o modelo pré-treinado
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' é a versão Nano; altere para 'yolov8s.pt' para a Small

# Treinar o modelo
model.train(
    data='dataset.yaml',  # Caminho do arquivo de configuração do dataset
    epochs=100,           # Número de épocas
    imgsz=640,            # Tamanho das imagens
    batch=16,             # Tamanho do batch
    name='yolov8_custom'  # Nome do experimento
)

# Avaliar o desempenho do modelo treinado
metrics = model.val()
print("Métricas do Modelo:", metrics)

# Fazer inferência em uma imagem de teste
results = model('caminho/para/imagem.jpg')  # Substitua pelo caminho da sua imagem
results.show()
