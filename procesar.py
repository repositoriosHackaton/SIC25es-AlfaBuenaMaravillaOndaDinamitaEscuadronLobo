import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from multiprocessing import Pool, cpu_count


mp_holistic = mp.solutions.holistic

def procesar_video(video_path):
    
    rel_path = os.path.relpath(video_path, "datos/clips")
    salida_path = os.path.join("datos/procesados", os.path.splitext(rel_path)[0] + ".npy")
    
 
    if os.path.exists(salida_path):
        print(f"Ya existe {salida_path}. Se omite el procesamiento.")
        return

    
    os.makedirs(os.path.dirname(salida_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video {video_path}")
        return

    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Video {video_path} no tiene frames válidos.")
        cap.release()
        return

    
    indices = np.linspace(0, total_frames - 1, num=20, dtype=int)


    frames_data = []

    
    with mp_holistic.Holistic(static_image_mode=True,
                              model_complexity=1,
                              enable_segmentation=False,
                              refine_face_landmarks=True) as holistic:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Frame {idx} no pudo ser leído en {video_path}")
                continue

          
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

          
            frame_dict = {}

           
            if results.face_landmarks:
                try:
                    nose = results.face_landmarks.landmark[1]
                    frame_dict["face"] = [nose.x, nose.y, nose.z]
                except Exception as e:
                    frame_dict["face"] = [0, 0, 0]
            else:
                frame_dict["face"] = [0, 0, 0]

         
            if results.pose_landmarks:
                try:
                    left_shoulder = results.pose_landmarks.landmark[11]
                    right_shoulder = results.pose_landmarks.landmark[12]
                    centro_hombros = [
                        (left_shoulder.x + right_shoulder.x) / 2,
                        (left_shoulder.y + right_shoulder.y) / 2,
                        (left_shoulder.z + right_shoulder.z) / 2
                    ]
                    frame_dict["pose"] = centro_hombros
                except Exception as e:
                    frame_dict["pose"] = [0, 0, 0]
            else:
                frame_dict["pose"] = [0, 0, 0]

           
            if results.left_hand_landmarks:
                left_hand = []
                for i in range(21):
                    try:
                        lm = results.left_hand_landmarks.landmark[i]
                        left_hand.append([lm.x, lm.y, lm.z])
                    except Exception as e:
                        left_hand.append([0, 0, 0])
                frame_dict["left_hand"] = left_hand
            else:
                frame_dict["left_hand"] = [[0, 0, 0] for _ in range(21)]

           
            if results.right_hand_landmarks:
                right_hand = []
                for i in range(21):
                    try:
                        lm = results.right_hand_landmarks.landmark[i]
                        right_hand.append([lm.x, lm.y, lm.z])
                    except Exception as e:
                        right_hand.append([0, 0, 0])
                frame_dict["right_hand"] = right_hand
            else:
                frame_dict["right_hand"] = [[0, 0, 0] for _ in range(21)]

            frames_data.append(frame_dict)

    cap.release()


    try:
        np.save(salida_path, frames_data, allow_pickle=True)
        print(f"Procesado y guardado: {salida_path}")
    except Exception as e:
        print(f"Error al guardar {salida_path}: {e}")

def obtener_videos():
    
    pattern = os.path.join("datos", "clips", "**", "*.mp4")
    return glob.glob(pattern, recursive=True)

if __name__ == "__main__":
    videos = obtener_videos()
    print(f"Se encontraron {len(videos)} videos para procesar.")

   
    n_cores = cpu_count()
    print(f"Usando {n_cores} núcleos de CPU para el procesamiento.")

    with Pool(processes=n_cores) as pool:
        pool.map(procesar_video, videos)
