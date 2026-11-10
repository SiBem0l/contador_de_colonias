import os
import glob

# Ruta a la carpeta donde están TODOS tus archivos .txt de etiquetas
# (Asegúrate de hacerlo tanto para la carpeta train como para val)
rutas_etiquetas = ["dataset_yolo/labels/train/*.txt", "dataset_yolo/labels/val/*.txt"]

for ruta in rutas_etiquetas:
    archivos_txt = glob.glob(ruta)
    
    for archivo in archivos_txt:
        with open(archivo, 'r') as f:
            lineas = f.readlines()
            
        nuevas_lineas = []
        for linea in lineas:
            partes = linea.strip().split()
            if len(partes) >= 5:
                # Cambiar el ID original por '0' y mantener las coordenadas
                partes[0] = '0' 
                nueva_linea = " ".join(partes) + "\n"
                nuevas_lineas.append(nueva_linea)
                
        # Sobrescribir el archivo con la clase unificada
        with open(archivo, 'w') as f:
            f.writelines(nuevas_lineas)

print("Todas las etiquetas han sido convertidas a 0")