from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Ruta corregida del modelo
ruta_modelo = "./Interfaz Para Movil/models_movil/best110.pt"

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"El modelo no se encontró en: {ruta_modelo}")

# Cargar el modelo entrenado
model = YOLO(ruta_modelo)

# Diccionario de rutas de imágenes
ruta_imagenes = "D:/InkaRoca-Deep-Learning/Interfaz Para Monitores/img/interface/"
imagenes = {
    0: {"producto": ruta_imagenes + "llavero.png", "info": ruta_imagenes + "llavero_inf.png"},
    1: {"producto": ruta_imagenes + "chompa.png", "info": ruta_imagenes + "chompa_inf.png"},
    2: {"producto": ruta_imagenes + "guantes.png", "info": ruta_imagenes + "guantes_inf.png"},
    3: {"producto": ruta_imagenes + "gorro.png", "info": ruta_imagenes + "gorro_inf.png"}
}

class CamaraApp(App):
    def build(self):
        # Diseño de la interfaz
        self.layout = BoxLayout(orientation='vertical')

        # Sección de la cámara (mitad superior)
        self.camera_view = Image(size_hint=(1, 0.6))
        self.layout.add_widget(self.camera_view)

        # Sección de imágenes detectadas
        self.layout_images = BoxLayout(orientation='horizontal', size_hint=(1, 0.4))
        self.image_producto = Image(size_hint=(0.5, 1))
        self.image_info = Image(size_hint=(0.5, 1))
        self.layout_images.add_widget(self.image_producto)
        self.layout_images.add_widget(self.image_info)
        self.layout.add_widget(self.layout_images)

        # Iniciar la cámara
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Actualizar cada 1/30 segundos

        return self.layout

    def update(self, dt):
        # Capturar frame de la cámara
        ret, frame = self.cap.read()
        if ret:
            # Procesar el frame con el modelo YOLO
            results = model(frame, stream=True, verbose=False)

            # Dibujar la detección en el frame
            for res in results:
                cajas = res.boxes
                for caja in cajas:
                    x1, y1, x2, y2 = [int(val) for val in caja.xyxy[0]]
                    clase = int(caja.cls[0])

                    if clase in imagenes:
                        self.image_producto.source = imagenes[clase]["producto"]
                        self.image_info.source = imagenes[clase]["info"]

                    # Dibujar la caja en el frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convertir el frame a textura para mostrarlo en Kivy
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_view.texture = texture

    def on_stop(self):
        # Liberar la cámara al cerrar la aplicación
        self.cap.release()

if __name__ == '__main__':
    CamaraApp().run()
