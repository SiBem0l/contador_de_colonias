import os
import shutil
import random

# 1. Configuración de carpetas origen
# Asegúrate de que estas carpetas existan y tengan tus 400 fotos y 400 txt
ruta_images = "dataset/images" 
ruta_labels = "dataset/labels" 

# 2. Configuración de carpetas destino (se crearán automáticamente)
dataset_dir = "dataset_yolo"
train_images = f"{dataset_dir}/images/train"
val_images = f"{dataset_dir}/images/val"
train_labels = f"{dataset_dir}/labels/train"
val_labels = f"{dataset_dir}/labels/val"

# Crear la estructura de directorios
for carpeta in [train_images, val_images, train_labels, val_labels]:
    os.makedirs(carpeta, exist_ok=True)

# 3. Leer todas las imágenes y mezclarlas
# Filtramos solo archivos de imagen por si hay otros archivos ocultos
imagenes = [f for f in os.listdir(ruta_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Semilla para que el resultado sea reproducible (opcional)
random.seed(42) 
random.shuffle(imagenes)

# 4. Calcular el corte (80% train, 20% val)
corte = int(len(imagenes) * 0.8)
lista_train = imagenes[:corte]
lista_val = imagenes[corte:]

# Función para copiar las imágenes y buscar su TXT correspondiente
def copiar_archivos(lista, dest_img, dest_lbl):
    for img_name in lista:
        # Rutas de la imagen original y destino
        origen_img = os.path.join(ruta_images, img_name)
        destino_img = os.path.join(dest_img, img_name)
        
        # El TXT tiene el mismo nombre que la imagen
        nombre_base = os.path.splitext(img_name)[0]
        txt_name = f"{nombre_base}.txt"
        
        origen_txt = os.path.join(ruta_labels, txt_name)
        destino_txt = os.path.join(dest_lbl, txt_name)
        
        # Copiar imagen
        shutil.copy(origen_img, destino_img)
        
        # Copiar etiqueta (solo si existe, previene errores)
        if os.path.exists(origen_txt):
            shutil.copy(origen_txt, destino_txt)
        else:
            print(f"Advertencia: No se encontró el txt para {img_name}")

# 5. Ejecutar la copia
print(f"Total de imágenes encontradas: {len(imagenes)}")
print(f"Copiando {len(lista_train)} archivos a Train...")
copiar_archivos(lista_train, train_images, train_labels)

print(f"Copiando {len(lista_val)} archivos a Val...")
copiar_archivos(lista_val, val_images, val_labels)

print("\n¡Listo! Tu dataset está preparado en la carpeta 'dataset_yolo'")