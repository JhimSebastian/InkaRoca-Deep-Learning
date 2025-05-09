from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Directorio base
directorio_base = os.path.dirname(os.path.abspath(__file__))

# Ruta del modelo
ruta_modelo = os.path.join(directorio_base, "models_movil", "best110.pt")

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"El modelo no se encontró en: {ruta_modelo}")

# Cargar el modelo YOLO
model = YOLO(ruta_modelo)

# Rutas de imágenes
imagenes = {
    0: {"producto": "img_movil/llavero.png", "info": "img_movil/llavero_inf.png"},
    1: {"producto": "img_movil/chompa.png", "info": "img_movil/chompa_inf.png"},
    2: {"producto": "img_movil/guantes.png", "info": "img_movil/guantes_inf.png"},
    3: {"producto": "img_movil/gorro.png", "info": "img_movil/gorro_inf.png"},
    4: {"producto": "img_movil/chalina.png", "info": "img_movil/chalina_inf.png"},
    5: {"producto": "img_movil/poncho.png", "info": "img_movil/poncho_inf.png"},
    6: {"producto": "img_movil/alfombra.png", "info": "img_movil/alfombra_inf.png"},
    7: {"producto": "img_movil/tapiz.png", "info": "img_movil/tapiz_inf.png"},
    8: {"producto": "img_movil/manta.png", "info": "img_movil/manta_inf.png"},
    9: {"producto": "img_movil/frazada.png", "info": "img_movil/frazada_inf.png"},
    10: {"producto": "img_movil/cubre cama.png", "info": "img_movil/cubre cama_inf.png"},
    11: {"producto": "img_movil/camino de meza.png", "info": "img_movil/camino de meza_inf.png"},
    12: {"producto": "img_movil/bolso.png", "info": "img_movil/bolso_inf.png"},
    13: {"producto": "img_movil/cartera.png", "info": "img_movil/cartera_inf.png"},
    14: {"producto": "img_movil/cartuchera.png", "info": "img_movil/cartuchera_inf.png"},
    15: {"producto": "img_movil/mochila.png", "info": "img_movil/mochila_inf.png"},
    16: {"producto": "img_movil/cojin.png", "info": "img_movil/cojin_inf.png"},
    17: {"producto": "img_movil/capa.png", "info": "img_movil/capa_inf.png"},
    18: {"producto": "img_movil/ruana.png", "info": "img_movil/ruana_inf.png"},
    19: {"producto": "img_movil/chaleco.png", "info": "img_movil/chaleco_inf.png"},
    20: {"producto": "img_movil/muñeca.png", "info": "img_movil/muñeca_inf.png"},
    21: {"producto": "img_movil/telarcito.png", "info": "img_movil/telarcito_inf.png"},
    22: {"producto": "img_movil/cuellera.png", "info": "img_movil/cuellera_inf.png"},
    23: {"producto": "img_movil/scarpin.png", "info": "img_movil/scarpin_inf.png"},
    24: {"producto": "img_movil/medias.png", "info": "img_movil/medias_inf.png"},
    25: {"producto": "img_movil/vincha.png", "info": "img_movil/vincha_inf.png"},
    26: {"producto": "img_movil/gancho.png", "info": "img_movil/gancho_inf.png"},
    27: {"producto": "img_movil/monedero.png", "info": "img_movil/monedero_inf.png"},
    28: {"producto": "img_movil/boina.png", "info": "img_movil/boina_inf.png"},
    29: {"producto": "img_movil/chall.png", "info": "img_movil/chall_inf.png"}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteccion', methods=['POST'])
def deteccion():
    try:
        data = request.json
        image_data = data['image'].split(",")[1]  # Remover encabezado de base64
        image_bytes = base64.b64decode(image_data)
        image_np = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        results = model(frame, stream=True, verbose=False)
        detecciones = []

        for res in results:
            for caja in res.boxes:
                x1, y1, x2, y2 = [int(val) for val in caja.xyxy[0]]
                clase = int(caja.cls[0])

                detecciones.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "clase": clase,
                    "producto": imagenes.get(clase, {}).get("producto", ""),
                    "info": imagenes.get(clase, {}).get("info", "")
                })

        return jsonify({"detecciones": detecciones})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
