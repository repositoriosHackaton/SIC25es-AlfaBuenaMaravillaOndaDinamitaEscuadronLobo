# preparar_abc.py
import os
import shutil

def preparar_datos(ruta_origen="datosABC/alfabeto_p",
                   ruta_destino_train="datosABC/train",
                   ruta_destino_val="datosABC/val",
                   max_imagenes_por_letra=1300,
                   split_ratio=0.8):
   
    clases = [d for d in os.listdir(ruta_origen) if os.path.isdir(os.path.join(ruta_origen, d))]
    print("Clases encontradas:", clases)

   
    counts = {}
    for clase in clases:
        carpeta = os.path.join(ruta_origen, clase)
        imagenes = sorted([f for f in os.listdir(carpeta) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        usable = min(len(imagenes), max_imagenes_por_letra)
        counts[clase] = usable
    min_count = min(counts.values())
    print(f"Usando {min_count} imágenes por clase para balancear el dataset.")

   
    os.makedirs(ruta_destino_train, exist_ok=True)
    os.makedirs(ruta_destino_val, exist_ok=True)

    
    for clase in clases:
        carpeta_origen = os.path.join(ruta_origen, clase)
        imagenes = sorted([f for f in os.listdir(carpeta_origen) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        imagenes = imagenes[:min_count]  

        split_index = int(len(imagenes) * split_ratio)
        train_imgs = imagenes[:split_index]
        val_imgs = imagenes[split_index:]

        train_dest = os.path.join(ruta_destino_train, clase)
        val_dest = os.path.join(ruta_destino_val, clase)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(val_dest, exist_ok=True)

        for img in train_imgs:
            origen = os.path.join(carpeta_origen, img)
            destino = os.path.join(train_dest, img)
            shutil.copy2(origen, destino)

        for img in val_imgs:
            origen = os.path.join(carpeta_origen, img)
            destino = os.path.join(val_dest, img)
            shutil.copy2(origen, destino)

        print(f"Clase '{clase}': {len(train_imgs)} train, {len(val_imgs)} val.")

def main():
    preparar_datos()

if __name__ == "__main__":
    main()
