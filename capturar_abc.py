import cv2
import mediapipe as mp
import time
import os
import threading


def obtener_siguiente_indice(carpeta, senia):
    archivos = [f for f in os.listdir(carpeta) if f.startswith(senia) and f.endswith(".jpg")]
    if not archivos:
        return 1
    numeros = []
    for f in archivos:
        try:
            numero = int(f.replace(senia, "").replace(".jpg", ""))
            numeros.append(numero)
        except:
            pass
    if not numeros:
        return 1
    return max(numeros) + 1


def preview_and_capture(senia, capture_duration=60):
   
    output_base = "datosABC/alfabeto_p"
    output_folder = os.path.join(output_base, senia)
    os.makedirs(output_folder, exist_ok=True)
    next_idx = obtener_siguiente_indice(output_folder, senia)

    
    margin = 25

    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Abrir la cámara.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

   
    start_capture_event = threading.Event()

    
    def wait_for_enter():
        input("PREVIEW: Ajusta tu mano y observa el recuadro en la ventana.\nCuando estés listo, presiona ENTER en la terminal para iniciar la captura...")
        start_capture_event.set()

    
    input_thread = threading.Thread(target=wait_for_enter)
    input_thread.daemon = True
    input_thread.start()

   
    print("Mostrando vista previa. Esperando confirmación en terminal para iniciar la captura...")
    while not start_capture_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        left_hand_landmarks = None
        if results.multi_hand_landmarks and results.multi_handedness:
         
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Left":
                    left_hand_landmarks = hand_landmarks
                    break

        
        if left_hand_landmarks is not None:
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in left_hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in left_hand_landmarks.landmark]
            x_min = max(0, min(x_coords) - margin)
            y_min = max(0, min(y_coords) - margin)
            x_max = min(w, max(x_coords) + margin)
            y_max = min(h, max(y_coords) + margin)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "PREVIEW", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Mano izquierda no detectada", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Vista Previa - Mano Izquierda", frame)
        cv2.waitKey(1)

  
    print("Iniciando captura de frames durante {} segundos...".format(capture_duration))
    capture_start = time.time()
    while time.time() - capture_start < capture_duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        left_hand_landmarks = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Left":
                    left_hand_landmarks = hand_landmarks
                    break

      
        if left_hand_landmarks is not None:
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in left_hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in left_hand_landmarks.landmark]
            x_min = max(0, min(x_coords) - margin)
            y_min = max(0, min(y_coords) - margin)
            x_max = min(w, max(x_coords) + margin)
            y_max = min(h, max(y_coords) + margin)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, "Capturando...", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            hand_crop = frame[y_min:y_max, x_min:x_max]
            try:
                hand_resized = cv2.resize(hand_crop, (64, 64))
            except Exception as e:
                print("Error al redimensionar:", e)
                continue
            filename = os.path.join(output_folder, f"{senia}{next_idx}.jpg")
            cv2.imwrite(filename, hand_resized)
            next_idx += 1

        cv2.imshow("Captura - Mano Izquierda", frame)
        cv2.waitKey(1)

    print("✅ Captura finalizada para la seña '{}'.".format(senia))
    cap.release()
    cv2.destroyAllWindows()
    input_thread.join()


while True:
    seña_input = input("Ingresa el nombre de la seña a capturar (o escribe SALIR para terminar): ").strip()
    if seña_input.upper() == "SALIR":
        print("Terminando el programa.")
        break
    preview_and_capture(seña_input, capture_duration=60)
