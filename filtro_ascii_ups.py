import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from PIL import Image, ImageFont, ImageDraw

# Kernel CUDA
kernel_code = """
__global__ void smooth(float *img_in, float *img_out, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    float sum = 0;
    int count = 0;
    for(int dx=-1; dx<=1; dx++) {
        for(int dy=-1; dy<=1; dy++) {
            int nx = x + dx;
            int ny = y + dy;
            if(nx >= 0 && ny >= 0 && nx < width && ny < height) {
                sum += img_in[ny * width + nx];
                count++;
            }
        }
    }
    img_out[y * width + x] = sum / count;
}
"""

def apply_ascii_ups_filter(img_bgr: np.ndarray, logo_pil: Image.Image) -> np.ndarray:
    # Crear contexto CUDA
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    
    try:
        mod = SourceModule(kernel_code)
        smooth = mod.get_function("smooth")

        # Redimensionar imagen a 240x120 para que quede igual que antes
        resized = cv2.resize(img_bgr, (240, 120))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        height, width = gray.shape

        # Reservar memoria en GPU
        img_gpu = cuda.mem_alloc(gray.nbytes)
        result_gpu = cuda.mem_alloc(gray.nbytes)
        cuda.memcpy_htod(img_gpu, gray)

        block = (16, 16, 1)
        grid = ((width + 15) // 16, (height + 15) // 16)

        smooth(img_gpu, result_gpu, np.int32(width), np.int32(height), block=block, grid=grid)

        smoothed = np.empty_like(gray)
        cuda.memcpy_dtoh(smoothed, result_gpu)
        smoothed_uint8 = smoothed.astype(np.uint8)

        # ASCII mapping
        ascii_chars = "@%#*+=-:. "
        num_chars = len(ascii_chars)
        scaled = (smoothed_uint8 / 255.0 * (num_chars - 1)).astype(np.uint8)
        ascii_image = np.array([[ascii_chars[pix] for pix in row] for row in scaled])

        # Crear imagen PIL para texto ASCII
        font = ImageFont.load_default()
        bbox = font.getbbox("A")
        char_width, char_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img_out = Image.new("L", (char_width * width, char_height * height), color=255)
        draw = ImageDraw.Draw(img_out)

        for y in range(height):
            for x in range(width):
                draw.text((x * char_width, y * char_height), ascii_image[y][x], fill=0, font=font)

        # Preparar logo para el marco
        logo_scale = 6
        logo_width = char_width * logo_scale
        logo_height = char_height * logo_scale
        mini_logo = logo_pil.resize((logo_width, logo_height))

        # Crear imagen final con borde (marco)
        border_x, border_y = logo_width, logo_height
        new_width = img_out.width + 2 * border_x
        new_height = img_out.height + 2 * border_y
        final_img = Image.new("L", (new_width, new_height), color=255)
        final_img.paste(img_out, (border_x, border_y))

        # Pegar logo en marco
        for x in range(0, new_width, logo_width):
            final_img.paste(mini_logo, (x, 0))
            final_img.paste(mini_logo, (x, new_height - logo_height))
        for y in range(0, new_height, logo_height):
            final_img.paste(mini_logo, (0, y))
            final_img.paste(mini_logo, (new_width - logo_width, y))

        # Convertir imagen final (grayscale PIL) a BGR NumPy para Flask/OpenCV
        final_rgb = final_img.convert("RGB")
        final_np = np.array(final_rgb)
        final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)

        return final_bgr

    finally:
        ctx.pop()
        ctx.detach()
