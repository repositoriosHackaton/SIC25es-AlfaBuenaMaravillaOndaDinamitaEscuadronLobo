from flask import Flask, Response, send_from_directory, request, jsonify
import cv2
import time
import torch
import mediapipe as mp
import os
from PIL import Image
import torchvision.transforms as transforms
import threading
import numpy as np
import torch.nn as nn
from clases import CLASES

########################
#  FUNCIONES AUXILIARES
########################

def speak_text(text):
    """Ejecuta TTS en un hilo separado. No sintetiza 'No se pudo inferir'."""
    if text.strip().lower() == "no se pudo inferir":
        return
    def _speak():
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 130)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def extraer_landmarks(frame_rgb, holistic):
    results = holistic.process(frame_rgb)
    frame_dict = {}
    if results.face_landmarks:
        try:
            nose = results.face_landmarks.landmark[1]
            frame_dict["face"] = [nose.x, nose.y, nose.z]
        except:
            frame_dict["face"] = [0, 0, 0]
    else:
        frame_dict["face"] = [0, 0, 0]
    if results.pose_landmarks:
        try:
            ls = results.pose_landmarks.landmark[11]
            rs = results.pose_landmarks.landmark[12]
            centro_hombros = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2, (ls.z + rs.z) / 2]
            frame_dict["pose"] = centro_hombros
        except:
            frame_dict["pose"] = [0, 0, 0]
    else:
        frame_dict["pose"] = [0, 0, 0]
    if results.left_hand_landmarks:
        left_hand = []
        for i in range(21):
            lm = results.left_hand_landmarks.landmark[i]
            left_hand.append([lm.x, lm.y, lm.z])
        frame_dict["left_hand"] = left_hand
    else:
        frame_dict["left_hand"] = [[0, 0, 0] for _ in range(21)]
    if results.right_hand_landmarks:
        right_hand = []
        for i in range(21):
            lm = results.right_hand_landmarks.landmark[i]
            right_hand.append([lm.x, lm.y, lm.z])
        frame_dict["right_hand"] = right_hand
    else:
        frame_dict["right_hand"] = [[0, 0, 0] for _ in range(21)]
    return frame_dict

def convertir_a_vector(frame_dict):
    cara = np.array(frame_dict["face"], dtype=np.float32)
    pose = np.array(frame_dict["pose"], dtype=np.float32)
    mano_izq = np.array(frame_dict["left_hand"]).flatten().astype(np.float32)
    mano_der = np.array(frame_dict["right_hand"]).flatten().astype(np.float32)
    return np.concatenate([cara, pose, mano_izq, mano_der], axis=0)

def submuestrear_20_frames(lista_vectores):
    n = len(lista_vectores)
    if n == 0:
        return None
    indices = np.linspace(0, n - 1, 20, dtype=int)
    vectores_20 = [lista_vectores[i] for i in indices]
    return np.array(vectores_20, dtype=np.float32)

########################
#  MODELOS (LSTM y CNN)
########################

class ModeloLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.5):
        super(ModeloLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def predecir_signo(secuencia_20, model, CLASES, device):
    x = torch.from_numpy(secuencia_20).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).item()
    return CLASES[pred_idx]

class CNN_ABC(nn.Module):
    def __init__(self, num_classes):
        super(CNN_ABC, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def predecir_fingerspelling(roi_img, cnn_model, abc_transform, index_to_letra, device):
    roi_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    input_tensor = abc_transform(roi_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cnn_model(input_tensor)
        _, pred = torch.max(outputs, 1)
    return index_to_letra[pred.item()]

########################
#  CARGA DE MODELOS
########################

def cargar_modelos(device):
    # Ejemplo de clases para el modelo LSTM; reemplaza según tus datos reales
    
    
    lstm_model = ModeloLSTM(
        input_size=132, 
        hidden_size=256, 
        output_size=len(CLASES), 
        num_layers=2, 
        dropout=0.5
    ).to(device)
    modelo_lstm_path = os.path.join('modelos', 'modelo_lstm.pth')
    if os.path.exists(modelo_lstm_path):
        lstm_model.load_state_dict(torch.load(modelo_lstm_path, map_location=device))
    lstm_model.eval()

    source_folder = os.path.join('datosABC', 'alfabeto_p')
    if os.path.exists(source_folder):
        index_to_letra = sorted(
            d for d in os.listdir(source_folder)
            if os.path.isdir(os.path.join(source_folder, d))
        )
    else:
        index_to_letra = ["A", "B", "C"]
    cnn_model = CNN_ABC(num_classes=len(index_to_letra)).to(device)
    modelo_cnn_path = 'model_abc.pth'
    if os.path.exists(modelo_cnn_path):
        cnn_model.load_state_dict(torch.load(modelo_cnn_path, map_location=device))
    cnn_model.eval()

    return lstm_model, cnn_model, CLASES, index_to_letra

########################
#  INICIALIZACIÓN
########################

device = "cuda" if torch.cuda.is_available() else "cpu"
lstm_model, cnn_model, CLASES, index_to_letra = cargar_modelos(device)

abc_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True
)

# Variables globales de estado
roi_x, roi_y = 10, 70
roi_width, roi_height = 170, 170

estado = "esperando"
buffer_landmarks = []
FRAMES_SIN_MANOS_UMBRAL = 5
contador_sin_manos = 0
TIEMPO_MOSTRANDO_RESULTADO = 5.0
resultado_lstm = "..."
tiempo_inicio_resultado = 0

cnn_mode = False
cnn_entry_time = 0
ultimo_tiempo_cnn = 0
capture_trigger_time = 0
palabra_acumulada = ""
letra_actual = ""
tts_spoken_cnn = False
tts_spoken_lstm = False

# Variable para activar/desactivar inferencia
inference_enabled = True

def mano_en_roi(results, frame_width, frame_height, rx, ry, rw, rh):
    if results.left_hand_landmarks:
        x_coords = [int(lm.x * frame_width) for lm in results.left_hand_landmarks.landmark]
        y_coords = [int(lm.y * frame_height) for lm in results.left_hand_landmarks.landmark]
        if x_coords and y_coords:
            cx = int((min(x_coords) + max(x_coords)) / 2)
            cy = int((min(y_coords) + max(y_coords)) / 2)
            if (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh):
                return True
    if results.right_hand_landmarks:
        x_coords = [int(lm.x * frame_width) for lm in results.right_hand_landmarks.landmark]
        y_coords = [int(lm.y * frame_height) for lm in results.right_hand_landmarks.landmark]
        if x_coords and y_coords:
            cx = int((min(x_coords) + max(x_coords)) / 2)
            cy = int((min(y_coords) + max(y_coords)) / 2)
            if (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh):
                return True
    return False

def hay_manos(frame_dict):
    hay_mano_izq = any(any(pt != 0 for pt in punto) for punto in frame_dict["left_hand"])
    hay_mano_der = any(any(pt != 0 for pt in punto) for punto in frame_dict["right_hand"])
    return hay_mano_izq or hay_mano_der

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def gen_frames():
    global estado, buffer_landmarks, contador_sin_manos, resultado_lstm, tiempo_inicio_resultado
    global cnn_mode, cnn_entry_time, ultimo_tiempo_cnn, capture_trigger_time, palabra_acumulada, letra_actual
    global tts_spoken_cnn, tts_spoken_lstm, inference_enabled

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # Dibuja la ROI
        roi_color = (0, 255, 0)
        if time.time() - capture_trigger_time < 0.5:
            roi_color = (0, 255, 255)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), roi_color, 2)

        if not inference_enabled:
            cv2.putText(frame, "Inferencia desactivada", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            frame_dict = extraer_landmarks(frame_rgb, holistic)

            usar_cnn = mano_en_roi(results, frame_width, frame_height, roi_x, roi_y, roi_width, roi_height)

            if usar_cnn:
                if not cnn_mode:
                    cnn_mode = True
                    cnn_entry_time = time.time()
                    ultimo_tiempo_cnn = 0
                    palabra_acumulada = ""
                    letra_actual = ""
                    tts_spoken_cnn = False
                if time.time() - cnn_entry_time < 1:
                    cv2.putText(frame, "Preparando...", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    if time.time() - ultimo_tiempo_cnn >= 2:
                        margin = 25
                        if results.left_hand_landmarks:
                            x_coords = [int(lm.x * frame_width) for lm in results.left_hand_landmarks.landmark]
                            y_coords = [int(lm.y * frame_height) for lm in results.left_hand_landmarks.landmark]
                        elif results.right_hand_landmarks:
                            x_coords = [int(lm.x * frame_width) for lm in results.right_hand_landmarks.landmark]
                            y_coords = [int(lm.y * frame_height) for lm in results.right_hand_landmarks.landmark]
                        else:
                            x_coords, y_coords = [], []

                        if x_coords and y_coords:
                            x_min = max(roi_x, min(x_coords) - margin)
                            y_min = max(roi_y, min(y_coords) - margin)
                            x_max = min(roi_x + roi_width, max(x_coords) + margin)
                            y_max = min(roi_y + roi_height, max(y_coords) + margin)
                            if x_max - x_min <= 0 or y_max - y_min <= 0:
                                roi_img = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
                            else:
                                roi_img = frame[y_min:y_max, x_min:x_max]
                        else:
                            roi_img = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

                        letra_actual = predecir_fingerspelling(roi_img, cnn_model, abc_transform, index_to_letra, device)
                        palabra_acumulada += letra_actual
                        ultimo_tiempo_cnn = time.time()
                        capture_trigger_time = time.time()

                    cv2.putText(frame, f"{letra_actual}", (roi_x + 5, roi_y + roi_height - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(frame, f"Resultado: {palabra_acumulada}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                if cnn_mode:
                    if not tts_spoken_cnn and palabra_acumulada != "":
                        speak_text(palabra_acumulada)
                        tts_spoken_cnn = True
                    if time.time() - cnn_entry_time < TIEMPO_MOSTRANDO_RESULTADO:
                        cv2.putText(frame, f"Resultado: {palabra_acumulada}", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    else:
                        cnn_mode = False

                manos_detectadas = hay_manos(frame_dict)
                if estado == "esperando":
                    if manos_detectadas:
                        buffer_landmarks.clear()
                        contador_sin_manos = 0
                        estado = "grabando"
                        tts_spoken_lstm = False
                elif estado == "grabando":
                    if manos_detectadas:
                        vector_132 = convertir_a_vector(frame_dict)
                        buffer_landmarks.append(vector_132)
                        contador_sin_manos = 0
                    else:
                        contador_sin_manos += 1
                        if contador_sin_manos > FRAMES_SIN_MANOS_UMBRAL:
                            secuencia_20 = submuestrear_20_frames(buffer_landmarks)
                            if secuencia_20 is not None:
                                resultado_lstm = predecir_signo(secuencia_20, lstm_model, CLASES, device)
                            else:
                                resultado_lstm = "No se pudo inferir"
                            estado = "mostrando_resultado"
                            tiempo_inicio_resultado = time.time()
                elif estado == "mostrando_resultado":
                    if manos_detectadas:
                        buffer_landmarks.clear()
                        contador_sin_manos = 0
                        estado = "grabando"
                        tts_spoken_lstm = False
                    else:
                        if time.time() - tiempo_inicio_resultado > TIEMPO_MOSTRANDO_RESULTADO:
                            estado = "esperando"
                            resultado_lstm = "..."

                if estado == "mostrando_resultado":
                    if not tts_spoken_lstm and resultado_lstm != "..." and resultado_lstm.strip().lower() != "no se pudo inferir":
                        speak_text(resultado_lstm)
                        tts_spoken_lstm = True
                    msg = f"Resultado: {resultado_lstm} (LSTM)"
                    color = (255, 255, 255)
                elif estado == "grabando":
                    msg = "Analizando... (LSTM)"
                    color = (0, 0, 255)
                else:
                    msg = "Esperando manos... (LSTM)"
                    color = (200, 200, 200)
                cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer_img = cv2.imencode('.jpg', frame)
        frame_bytes = buffer_img.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

########################
#  RUTAS FLASK
########################

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_inference', methods=['POST'])
def toggle_inference():
    global inference_enabled
    inference_enabled = not inference_enabled
    return jsonify({"inference_enabled": inference_enabled})

@app.route('/modelos')
def serve_modelos():
    return send_from_directory('.', 'modelos.html')

@app.route('/proyecto')
def serve_proyecto():
    return send_from_directory('.', 'proyecto.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)


########################
#  MAIN
########################

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cap.release()
        holistic.close()
