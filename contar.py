from ultralytics import YOLO

# 1. Cargar tu modelo entrenado
model = YOLO('best.pt')

# Definir la imagen y los niveles de confianza que deseas probar
imagen_prueba = 'a.jpg'
niveles_confianza = [0.2, 0.4, 0.6, 0.8]

print(f"Iniciando prueba con múltiples configuraciones de confianza en: {imagen_prueba}\n" + "-"*50)

# Bucle para iterar sobre cada nivel de confianza
for conf_actual in niveles_confianza:
    
    # 2. Hacer la predicción usando el conf_actual del bucle
    # verbose=False oculta el texto de procesamiento por consola para que el print final sea más limpio
    resultados = model.predict(source=imagen_prueba, conf=conf_actual, iou=0.5, verbose=False)
    resultado = resultados[0]
    
    # 3. Contar la cantidad de cajas detectadas
    cantidad_bacterias = len(resultado.boxes)
    print(f"Confianza configurada al {conf_actual*100:.0f}% (conf={conf_actual}): Se detectaron {cantidad_bacterias} colonias.")
    
    # 4. Guardar la imagen con un nombre dinámico para no sobrescribir la anterior
    nombre_archivo = f'resultado_conf_{conf_actual}.jpg'
    resultado.save(nombre_archivo)

print("-"*50 + "\n¡Prueba finalizada! Revisa tu carpeta para ver las diferentes imágenes guardadas.")