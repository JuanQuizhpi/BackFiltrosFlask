from flask import Flask, request, send_file
import numpy as np
from PIL import Image
import io
import cv2

# Filtros CUDA
from filtro_log import filtro_log_cuda
from filtro_gaussiano import filtro_gaussiano_cuda
from filtro_media import filtro_media_cuda
from filtro_cartoon import cartoon_filter 
from filtro_sketch import sketch_filter
from filtro_termico import apply_thermal_filter
from filtro_ascii_ups import apply_ascii_ups_filter

app = Flask(__name__)

@app.route('/procesar-imagen', methods=['POST'])
def procesar_imagen():
    # Verifica que se haya enviado la imagen y el tipo de filtro
    if 'imagen' not in request.files or 'tipoFiltro' not in request.form:
        return "Faltan datos en la solicitud", 400

    file = request.files['imagen']
    tipo_filtro = request.form['tipoFiltro']
    
    #Para los nuevos filtros
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return "No se pudo decodificar la imagen", 400

    # Abrimos la imagen
    image = Image.open(file.stream)
    
    #Abrimos la imagen del logo
    logo = Image.open("upslogo.png").convert("L")
    
    # Convertimos a escala de grises si hace falta
    image_gray = image.convert("L")
    image_np = np.array(image_gray)

    # Procesamos según el filtro solicitado
    if tipo_filtro == "filtroLog":
        imagen_resultante = filtro_log_cuda(image_np)
    elif tipo_filtro == "filtroMedia":
        imagen_resultante = filtro_media_cuda(image_np)
    elif tipo_filtro == "filtroGaussiano":
        imagen_resultante = filtro_gaussiano_cuda(image_np)
    elif tipo_filtro == "filtroCartoon":
        imagen_resultante = cartoon_filter(img)
    elif tipo_filtro == "filtroSketch":
        imagen_resultante = sketch_filter(img)
    elif tipo_filtro == "filtroTermico":
        imagen_resultante = apply_thermal_filter(img)
    elif tipo_filtro == "filtroAsciiUps":
        imagen_resultante = apply_ascii_ups_filter(img, logo)
    else:
        return "Filtro no reconocido", 400
    
    resultado_rgb = cv2.cvtColor(imagen_resultante, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(resultado_rgb)
    img_io = io.BytesIO()
    img_pil.save(img_io, format='PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
