from ultralytics import YOLO

# 1. Cargar un modelo preentrenado como base (esencial para que entrene rápido y bien)
model = YOLO('yolov8n.pt') # Usamos la versión 'nano', pero puedes usar 'yolov8s.pt' o mayores

# 2. Iniciar el entrenamiento
resultados = model.train(
    data='dataset_yolo/data.yaml', # Ruta absoluta o relativa a tu archivo yaml
    epochs=100,                  # Número de veces que el modelo verá todo el dataset
    imgsz=1024,                   # Tamaño al que se redimensionarán las imágenes (640 es estándar)
    batch=16,                    # Cuántas imágenes procesa a la vez (bájalo a 8 o 4 si te quedas sin memoria RAM/VRAM)
    name='primer_modelo' # Nombre de la carpeta donde se guardarán los resultados
)