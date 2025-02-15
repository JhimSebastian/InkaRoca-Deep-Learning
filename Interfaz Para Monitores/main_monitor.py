from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Ruta del modelo
ruta_modelo = "D:/InkaRoca-Deep-Learning/Interfaz Para Movil/models_movil/best110.pt"
if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"El modelo no se encontró en: {ruta_modelo}")

# Cargar el modelo YOLO
model = YOLO(ruta_modelo)

# Rutas de imágenes
ruta_imagenes = "D:/InkaRoca-Deep-Learning/Interfaz Para Movil/img_movil/"
imagenes = {
    0: {"producto": ruta_imagenes + "llavero.png", "info": ruta_imagenes + "llavero_inf.png"},
    1: {"producto": ruta_imagenes + "chompa.png", "info": ruta_imagenes + "chompa_inf.png"},
    2: {"producto": ruta_imagenes + "guantes.png", "info": ruta_imagenes + "guantes_inf.png"},
    3: {"producto": ruta_imagenes + "gorro.png", "info": ruta_imagenes + "gorro_inf.png"}
}

class CamaraApp(App):
    def build(self):
        # Layout principal con fondo
        self.layout = RelativeLayout()
        
        # Imagen de fondo
        self.background = Image(source=ruta_imagenes + "background.png", allow_stretch=True, keep_ratio=False)
        self.layout.add_widget(self.background)
        
        # Vista de cámara (cuadro superior grande)
        self.camera_view = Image(size_hint=(0.8, 0.4), pos_hint={"center_x": 0.5, "top": 0.85})
        self.layout.add_widget(self.camera_view)
        
        # Imagen de producto detectado (abajo izquierda)
        self.image_producto = Image(size_hint=(0.35, 0.35), pos_hint={"x": 0.1, "y": 0.1})
        self.layout.add_widget(self.image_producto)
        
        # Imagen de información (abajo derecha)
        self.image_info = Image(size_hint=(0.35, 0.35), pos_hint={"right": 0.9, "y": 0.1})
        self.layout.add_widget(self.image_info)
        
        # Iniciar la cámara
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
        return self.layout

    def update(self, dt):
        ret, frame = self.cap.read()
        if ret:
            results = model(frame, stream=True, verbose=False)
            
            for res in results:
                cajas = res.boxes
                for caja in cajas:
                    x1, y1, x2, y2 = [int(val) for val in caja.xyxy[0]]
                    clase = int(caja.cls[0])
                    
                    if clase in imagenes:
                        self.image_producto.source = imagenes[clase]["producto"]
                        self.image_info.source = imagenes[clase]["info"]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_view.texture = texture

    def on_stop(self):
        self.cap.release()

if __name__ == '__main__':
    CamaraApp().run()
