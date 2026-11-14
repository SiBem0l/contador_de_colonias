import cv2
import numpy as np

def preprocesar_placa(ruta_img, invertir=True):
    img = cv2.imread(ruta_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar temporalmente para detectar el círculo rápido y estable
    h, w = gray.shape
    scale = 800.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    small_blur = cv2.GaussianBlur(small, (9, 9), 2)

    circles = cv2.HoughCircles(
        small_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=small.shape[0] // 2,
        param1=50,
        param2=30,
        minRadius=int(min(small.shape) * 0.25),
        maxRadius=int(min(small.shape) * 0.5)
    )

    if circles is not None:
        cx, cy, r = (circles[0][0] / scale).astype(int)
        # Crear máscara circular para borrar la mesa y esquinas
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), int(r * 0.96), 255, -1)
        
        # Recortar bounding box de la placa
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        crop = cv2.bitwise_and(gray, gray, mask=mask)[y1:y2, x1:x2]
    else:
        crop = gray

    # Invertir luminancia: colonias oscuras -> puntos blancos sobre fondo oscuro
    if invertir:
        crop = cv2.bitwise_not(crop)

    # Mejorar contraste local (CLAHE) y volver a 3 canales para YOLO
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    crop = clahe.apply(crop)
    return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)