from ultralytics import YOLO

# 1. Cargar un modelo preentrenado como base (esencial para que entrene rápido y bien)
model = YOLO("yolo26n.pt")

# 2. Iniciar el entrenamiento apuntando a la ruta de Drive
# resultados = model.train(
#     data='/content/drive/MyDrive/COLAB/dataset_yolo/data.yaml', # Ruta en tu Google Drive
#     epochs=50,
#     imgsz=1024,
#     batch=8,
#     name='primer_modelo'
# )
model.train(
    data="/content/drive/MyDrive/COLAB/dataset_yolo/data.yaml",
    epochs=100,
    imgsz=1024,
    batch = 8,
    name='segundo_modelo',
    max_det=300,
    
    hsv_h=0.05,
    hsv_s=0.55,
    hsv_v=0.40,
    degrees=15,
    fliplr=0.5,
    flipud=0.5
)