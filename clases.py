import os

def obtener_clases(ruta_base="datos/procesados"):
    
    clases = sorted(
        [d for d in os.listdir(ruta_base)
         if os.path.isdir(os.path.join(ruta_base, d))]
    )
    return clases


CLASES = obtener_clases()
