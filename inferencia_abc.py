# inferencia_abc.py
import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import deque, Counter

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

def main():
    
    source_folder = "datosABC/alfabeto_p"
    index_to_letra = sorted([d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))])
    num_classes = len(index_to_letra)
    print("Clases encontradas:", index_to_letra)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_ABC(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("model_abc.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    margin = 25
   
    pred_buffer = deque(maxlen=5)

    print("Presiona ESC para salir.")

    while True:
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

        altura, ancho, _ = frame.shape
        letra_predicha = "N/A"

        if left_hand_landmarks is not None:
            x_coords = [int(lm.x * ancho) for lm in left_hand_landmarks.landmark]
            y_coords = [int(lm.y * altura) for lm in left_hand_landmarks.landmark]
            x_min = max(0, min(x_coords) - margin)
            y_min = max(0, min(y_coords) - margin)
            x_max = min(ancho, max(x_coords) + margin)
            y_max = min(altura, max(y_coords) + margin)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            mano_recortada = frame[y_min:y_max, x_min:x_max]
            mano_pil = Image.fromarray(cv2.cvtColor(mano_recortada, cv2.COLOR_BGR2RGB))
            mano_tensor = transform(mano_pil)
            mano_tensor = mano_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(mano_tensor)
                _, pred = torch.max(outputs, 1)
                pred_letter = index_to_letra[pred.item()]
                pred_buffer.append(pred_letter)
                # Suavizado: voto mayoritario en el buffer
                if len(pred_buffer) > 0:
                    most_common = Counter(pred_buffer).most_common(1)[0][0]
                    letra_predicha = most_common
                else:
                    letra_predicha = pred_letter

            cv2.putText(frame, f"Pred: {letra_predicha}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Mano izquierda no detectada", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Inferencia ABC - Mano Izquierda", frame)
        key = cv2.waitKey(1)
        if key == 27:  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
