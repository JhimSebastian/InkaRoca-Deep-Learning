# Librerias
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math

ventas = {
    'Llavero': 0,
    'Chompa': 0,
    'Guantes': 0,
    'Gorro': 0
}

def run() -> None:
    # Declarando variables globales
    global cap, seccion_video, model, nombres_clases, img_background, img_llavero, img_chompa, img_guantes, img_gorro, img_btn_comprar
    global img_llavero_inf, img_chompa_inf, img_guantes_inf, img_gorro_inf, pantalla
    global seccion_img_producto, seccion_img_producto_inf, seccion_img_producto_mas_vendido
    global clase
    
    # Ejecutamos Ventana principal
    ventana_principal()
    
    # Cargando el modelo YOLO
    model = YOLO("./models/best110.pt")

    # Clases: 0 -> llavero | 1 -> Chompa | 2 -> Guantes | 3 -> Gorro
    nombres_clases = ["Llavero", "Chompa", "Guantes", "Gorro"]

    # Cargando imagenes de productos con OpenCV
    img_llavero = cv2.imread("./img/interface/llavero.png")
    img_chompa = cv2.imread("./img/interface/chompa.png")
    img_guantes = cv2.imread("./img/interface/guantes.png")
    img_gorro = cv2.imread("./img/interface/gorro.png")

    # Cargando imagenes de información de productos con OpenCV
    img_llavero_inf = cv2.imread("./img/interface/llavero_inf.png")
    img_chompa_inf = cv2.imread("./img/interface/chompa_inf.png")
    img_guantes_inf = cv2.imread("./img/interface/guantes_inf.png")
    img_gorro_inf = cv2.imread("./img/interface/gorro_inf.png")

    # Iniciando la captura de video
    iniciar_video_camara()

    # Sección de imagen del producto en la ventana principal (63, 234)
    seccion_img_producto = Label(pantalla)
    seccion_img_producto.place(x=63, y=140)

    # Sección de imagen de información del producto en la ventana principal (1125, 234)
    seccion_img_producto_inf = Label(pantalla)
    seccion_img_producto_inf.place(x=1120, y=160)

    # Sección de imagen del producto más vendido en la ventana principal (63, 515)
    seccion_img_producto_mas_vendido = Label(pantalla)
    seccion_img_producto_mas_vendido.place(x=63, y=490)

    # Producto más vendido INICIAL
    producto_mas_vendio: str = max(ventas, key=ventas.get)

    match producto_mas_vendio:
        case "Llavero":
            img_producto_mas_vendido = img_llavero
        case "Chompa":
            img_producto_mas_vendido = img_chompa
        case "Guantes":
            img_producto_mas_vendido = img_guantes
        case "Gorro":
            img_producto_mas_vendido = img_gorro

    # Dibujando el producto más vendido
    dibujar_producto_mas_vendido(img_producto_mas_vendido)

    # Refrescar el video cada 10ms
    refrescar_video()

    # Loop de la ventana
    pantalla.mainloop()


def ventana_principal() -> None:
    # Inicializando ventana principal
    global pantalla, img_background, img_producto_mas_vendido, img_btn_comprar
    pantalla = Tk()
    pantalla.title("INKAROCA ML")
    #pantalla.geometry("1920x1080")
    pantalla.geometry("1440x810")

    # Background
    img_background = ImageTk.PhotoImage(Image.open("./img/interface/background.png"))
    background = Label(image=img_background, text="INKAROCA ML")
    background.place(x=0, y=0, relwidth=1, relheight=1)

    # Boton de compra
    img_btn_comprar = ImageTk.PhotoImage(Image.open("./img/interface/btn_comprar.png"))
    btn_comprar = Button(image=img_btn_comprar, command=lambda: compra_producto(clase))
    btn_comprar.place(x=1120, y=605)

def iniciar_video_camara() -> None:
    global cap, seccion_video, pantalla
    # Sección de videocamara en la ventana principal (521, 314)
    seccion_video = Label(pantalla)
    seccion_video.place(x=390, y=200)

    # Inicializando la captura de video
    # cap = cv2.VideoCapture(1)
    # Si no funciona para windows, usar
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Resolución de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 685)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 498)


def refrescar_video() -> None:
    global cap, seccion_video, model, nombres_clases, img_background, img_llavero, img_chompa, img_guantes, img_gorro
    global img_llavero_inf, img_chompa_inf, img_guantes_inf, img_gorro_inf, pantalla

    detect_guia = False

    # Leyendo el frame de la cámara si es valido
    if cap:
        ret, frame = cap.read()
        # Procesando el frame
        frame = cv2.flip(frame, 1) # Efecto espejo
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret:
            # Obteniendo datos del objeto detectado
            try:
                global clase
                x1, y1, x2, y2, clase, confidencia =  obteniendo_datos_del_objeto(frame)
                detect_guia = True
            except TypeError:
                x1, y1, x2, y2, clase, confidencia = None, None, None, None, None, 0

            # Dibujando la caja del objeto detectado, si la confidencia es mayor a 50%
            if confidencia > 50:
                dibujar_caja_del_objeto(x1, y1, x2, y2, clase, confidencia, frame)
            
            # Si no se detecta ningun objeto, quitamos la imagen del producto
            if not detect_guia:
                limpiar_producto()

            # Redimensionar el frame
            frame = imutils.resize(frame, width=685)

            # Convertir el video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Mostrando el video en la ventana
            seccion_video.configure(image=img)
            seccion_video.image = img
            seccion_video.after(10, refrescar_video) # Actualizar cada 10ms

        else:
            cap.release() # Liberar la cámara


def obteniendo_datos_del_objeto(frame) -> tuple:
    global cap, model, nombres_clases, img_llavero, img_chompa, img_guantes, img_gorro
    global img_llavero_inf, img_chompa_inf, img_guantes_inf, img_gorro_inf

    # Obteniendo los datos del objeto con el modelo YOLO
    results = model(frame, stream=True, verbose=False)

    # Recorriendo los resultados
    for res in results:
        # Extrayendo las cajas
        cajas = res.boxes
        # Recorriendo las cajas
        for caja in cajas:
            # Obteniendo las coordenadas de la caja en enteros
            x1, y1, x2, y2 = [int(val) for val in caja.xyxy[0]]

            # Corrigiendo error para objetos al limite de la captura
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > 914: x2 = 914
            if y2 > 665: y2 = 665

            # Obteniendo el nombre de la clase por el indice
            clase: str = nombres_clases[int(caja.cls[0])]

            # Obteniendo la confidencia en porcentaje
            confidencia: int = math.ceil(caja.conf[0] * 100)
            
            return x1, y1, x2, y2, clase, confidencia
        
def dibujar_caja_del_objeto(x1, y1, x2, y2, clase, confidencia, frame) -> None:
    # Clasificando el objeto
    match clase:
        case "Llavero":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (237, 64, 61), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando el llavero
            dibujar_producto(img_llavero, img_llavero_inf)
        case "Chompa":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (253, 154, 143), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando la chompa
            dibujar_producto(img_chompa, img_chompa_inf)
        case "Guantes":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (252, 113, 34), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando los guantes
            dibujar_producto(img_guantes, img_guantes_inf)
        case "Gorro":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (242, 176, 34), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando el gorro
            dibujar_producto(img_gorro, img_gorro_inf)


def dibujar_producto(img_producto, img_producto_inf) -> None:
    img = img_producto
    inf = img_producto_inf

    # Colocando la imagen del producto en la sección de imagen
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    seccion_img_producto.configure(image=img_)
    seccion_img_producto.image = img_

    # Colocando la imagen de información del producto en la sección de información
    inf = np.array(inf, dtype=np.uint8)
    inf = cv2.cvtColor(inf, cv2.COLOR_RGB2BGR)
    inf = Image.fromarray(inf)

    inf_ = ImageTk.PhotoImage(image=inf)
    seccion_img_producto_inf.configure(image=inf_)
    seccion_img_producto_inf.image = inf_

def dibujar_producto_mas_vendido(img_producto_mas_vendido: str) -> None:
    img = img_producto_mas_vendido

    # Colocando la imagen del producto en la sección de imagen
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    seccion_img_producto_mas_vendido.configure(image=img_)
    seccion_img_producto_mas_vendido.image = img_

def limpiar_producto() -> None:
    # Limpiando la imagen del producto
    seccion_img_producto.config(image="")
    seccion_img_producto_inf.config(image="")

def compra_producto(producto: str) -> None:
    try:
        # Aumentando la cantidad de ventas del producto
        ventas[producto] += 1
    except KeyError:
        messagebox.showerror("Error", "No se ha detectado ningún producto")
        return

    # Producto más vendido
    producto_mas_vendio: str = max(ventas, key=ventas.get)

    match producto_mas_vendio:
        case "Llavero":
            img_producto_mas_vendido = img_llavero
        case "Chompa":
            img_producto_mas_vendido = img_chompa
        case "Guantes":
            img_producto_mas_vendido = img_guantes
        case "Gorro":
            img_producto_mas_vendido = img_gorro

    # Dibujando el producto más vendido
    dibujar_producto_mas_vendido(img_producto_mas_vendido)

    messagebox.showinfo("¡Comprado!", f"Se ha comprado un {producto}!")

if __name__ == "__main__":
    run()