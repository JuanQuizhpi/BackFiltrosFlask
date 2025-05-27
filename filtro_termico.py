import numpy as np
import cv2
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Manejo de contexto CUDA
def create_cuda_context():
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    return ctx

def cleanup_cuda_context(ctx):
    ctx.pop()
    ctx.detach()

# Kernel CUDA para mapa térmico estilo "jet"
kernel_code = """
__device__ void map_to_thermal(unsigned char gray, unsigned char* r, unsigned char* g, unsigned char* b) {
    float value = gray / 255.0f;

    if (value <= 0.25f) {
        *r = 0;
        *g = (unsigned char)(value * 4 * 255);
        *b = 255;
    } else if (value <= 0.5f) {
        *r = 0;
        *g = 255;
        *b = (unsigned char)((1.0f - (value - 0.25f) * 4) * 255);
    } else if (value <= 0.75f) {
        *r = (unsigned char)((value - 0.5f) * 4 * 255);
        *g = 255;
        *b = 0;
    } else {
        *r = 255;
        *g = (unsigned char)((1.0f - (value - 0.75f) * 4) * 255);
        *b = 0;
    }
}

__global__ void thermal_filter(unsigned char *gray, unsigned char *output,
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx_gray = y * width + x;
    int idx_out = idx_gray * 3;

    unsigned char r, g, b;
    map_to_thermal(gray[idx_gray], &r, &g, &b);

    output[idx_out] = r;
    output[idx_out + 1] = g;
    output[idx_out + 2] = b;
}
"""

def apply_thermal_filter(img):
    ctx = create_cuda_context()  # Crear contexto CUDA
    try:
        mod = SourceModule(kernel_code)
        thermal_filter = mod.get_function("thermal_filter")

        # Convertir imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        gray_flat = gray.flatten()
        output_flat = np.empty((h * w * 3), dtype=np.uint8)

        # Reservar memoria en GPU
        gray_gpu = cuda.mem_alloc(gray_flat.nbytes)
        output_gpu = cuda.mem_alloc(output_flat.nbytes)

        # Copiar a GPU
        cuda.memcpy_htod(gray_gpu, gray_flat)

        # Configurar bloques e hilos
        block = (16, 16, 1)
        grid = ((w + 15) // 16, (h + 15) // 16)

        # Ejecutar kernel
        thermal_filter(gray_gpu, output_gpu, np.int32(w), np.int32(h), block=block, grid=grid)

        # Recuperar datos
        cuda.memcpy_dtoh(output_flat, output_gpu)
        result = output_flat.reshape((h, w, 3))

        return result

    finally:
        cleanup_cuda_context(ctx)  # Liberar contexto CUDA
