import cv2
import numpy as np
import torch
import mediapipe as mp
import time


from clases import CLASES



import torch.nn as nn

class ModeloLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
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



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo usado: {device}")

model = ModeloLSTM(
    input_size=132, 
    hidden_size=256,
    output_size=len(CLASES),
    num_layers=2,
    dropout=0.5
).to(device)

ruta_modelo = "modelos/modelo_lstm.pth"
model.load_state_dict(torch.load(ruta_modelo, map_location=device))
model.eval()


mp_holistic = mp.solutions.holistic


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
            centro_hombros = [
                (ls.x + rs.x) / 2,
                (ls.y + rs.y) / 2,
                (ls.z + rs.z) / 2
            ]
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

def predecir_signo(secuencia_20):
    x = torch.from_numpy(secuencia_20).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).item()
    return CLASES[pred_idx]


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True
    )

    estado = "esperando"   
    buffer_landmarks = []
    resultado_actual = "..."
    
    
    FRAMES_SIN_MANOS_UMBRAL = 5
    contador_sin_manos = 0
    
  
    TIEMPO_MOSTRANDO_RESULTADO = 5.0  
    tiempo_inicio_resultado = None

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_dict = extraer_landmarks(frame_rgb, holistic)

        hay_mano_izq = any(any(pt != 0 for pt in punto) for punto in frame_dict["left_hand"])
        hay_mano_der = any(any(pt != 0 for pt in punto) for punto in frame_dict["right_hand"])
        manos_detectadas = hay_mano_izq or hay_mano_der

     
        if estado == "esperando":
            if manos_detectadas:
                buffer_landmarks.clear()
                contador_sin_manos = 0
                estado = "grabando"

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
                        prediccion = predecir_signo(secuencia_20)
                        resultado_actual = prediccion
                    else:
                        resultado_actual = "No se pudo inferir"
                    
                    estado = "mostrando_resultado"
                    tiempo_inicio_resultado = time.time()

        elif estado == "mostrando_resultado":
           
            if manos_detectadas:
                buffer_landmarks.clear()
                contador_sin_manos = 0
                estado = "grabando"
            else:
                
                if tiempo_inicio_resultado is not None:
                    if (time.time() - tiempo_inicio_resultado) > TIEMPO_MOSTRANDO_RESULTADO:
                        estado = "esperando"
                        resultado_actual = "..."  

     
        if estado == "grabando":
            msg = "Analizando..."
            color = (0, 0, 255)
        elif estado == "mostrando_resultado":
            msg = f"Resultado: {resultado_actual}"
            color = (255, 0, 0)
        else:
            msg = "Esperando manos..."
            color = (200, 200, 200)

        cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Inferencia en tiempo real", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    holistic.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
