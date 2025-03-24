import os
import cv2
import time


ruta_videos = "datos/clips/"


FPS = 30
DURACION_SEGUNDOS = 2   
FRAMES_TOTALES = FPS * DURACION_SEGUNDOS  
RESOLUCION = (1280, 720)


def obtener_siguiente_indice(ruta_carpeta, nombre_seña):
    archivos = [f for f in os.listdir(ruta_carpeta) if f.startswith(f"{nombre_seña}_") and f.endswith(".mp4")]
    if not archivos:
        return 0
    numeros = sorted([int(f.replace(f"{nombre_seña}_", "").replace(".mp4", "")) for f in archivos])
    return numeros[-1] + 1 


def capturar_video(nombre_seña, indice):
    carpeta_seña = os.path.join(ruta_videos, nombre_seña)
    os.makedirs(carpeta_seña, exist_ok=True)

    archivo_salida = os.path.join(carpeta_seña, f"{nombre_seña}_{indice}.mp4")

    cap = cv2.VideoCapture(0)
    cap.set(3, RESOLUCION[0])
    cap.set(4, RESOLUCION[1])

   
    for _ in range(10):  
        cap.read()

    
    time.sleep(0.5)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara.")
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(archivo_salida, fourcc, FPS, RESOLUCION)

    print(f"🎥 Grabando {nombre_seña} -> {archivo_salida}")

    
    for i in range(FRAMES_TOTALES):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow(f"Grabando {nombre_seña}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print("❌ Captura cancelada por el usuario.")
            return False  

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Captura finalizada con {FRAMES_TOTALES} frames: {archivo_salida}")
    return True  


def capturar_clips(nombre_seña, num_clips):
    carpeta_seña = os.path.join(ruta_videos, nombre_seña)
    os.makedirs(carpeta_seña, exist_ok=True)

    indice = obtener_siguiente_indice(carpeta_seña, nombre_seña)

    print("\n⌛ Preparándose para la grabación...")
    for i in range(3, 0, -1):  
        print(f"⌛ Iniciando en {i}...")
        time.sleep(1)

    print(f"🎯 Capturando {num_clips} clips de '{nombre_seña}'...")

    for _ in range(num_clips):
        if not capturar_video(nombre_seña, indice):
            break  
        indice += 1  


def menu_captura():
    while True:
        nombre_seña = input("\n📝 Ingrese el nombre de la seña (o 'salir' para terminar): ").strip()
        if nombre_seña.lower() == "salir":
            break

        while True:
            try:
                num_clips = int(input(f"🎬 ¿Cuántos clips deseas capturar para '{nombre_seña}'? "))
                if num_clips <= 0:
                    print("⚠️ Ingresa un número válido mayor que 0.")
                    continue
                break
            except ValueError:
                print("⚠️ Ingresa un número válido.")

        capturar_clips(nombre_seña, num_clips)


if __name__ == "__main__":
    print("🎬 Captura de datos para Lengua de Señas")
    menu_captura()
    print("🚀 Finalizado.")
