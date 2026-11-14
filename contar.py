from ultralytics import YOLO
from preprocess_imagen import preprocesar_placa

# 1. Cargar tu modelo entrenado
model = YOLO('best_3.pt')

# Preprocesar la imagen (recorte 1:1 + grises invertidos + CLAHE)
nombre_img = 'a5.jpg'
imagen_prueba = preprocesar_placa(nombre_img)
niveles_confianza = [0.05, 0.1, 0.2, 0.4, 0.6, 0.65]

print(f"Iniciando prueba con múltiples configuraciones de confianza en: {nombre_img}\n" + "-"*50)

for conf_actual in niveles_confianza:
    # 2. Pasar max_det=1500 e imgsz=1024 explícitamente
    resultados = model.predict(
        source=imagen_prueba,
        imgsz=1024,
        conf=conf_actual,
        iou=0.5,
        max_det=1500,
        verbose=False
    )
    resultado = resultados[0]
    
    # 3. Contar la cantidad real de cajas detectadas
    cantidad_bacterias = len(resultado.boxes)
    print(f"Confianza configurada al {conf_actual*100:.0f}% (conf={conf_actual}): Se detectaron {cantidad_bacterias} colonias.")
    
    # 4. Guardar la imagen resultante
    nombre_archivo = f'resultado_conf_{conf_actual}.jpg'
    resultado.save(nombre_archivo, labels=False, line_width=1)

print("-"*50 + "\n¡Prueba finalizada! Revisa tu carpeta para ver las diferentes imágenes guardadas.")