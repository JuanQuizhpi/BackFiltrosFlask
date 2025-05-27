import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit 
from pycuda.compiler import SourceModule
import math

# Inicialización manual del contexto CUDA
def create_cuda_context():
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    return ctx

def cleanup_cuda_context(ctx):
    ctx.pop()
    ctx.detach()

# Código CUDA
kernel_code = """
__constant__ float sobel_x[9] = { -1, 0, 1,
                                  -2, 0, 2,
                                  -1, 0, 1 };

__constant__ float sobel_y[9] = { -1, -2, -1,
                                   0,  0,  0,
                                   1,  2,  1 };

__global__ void sobel_filter(const float *img, float *output, int width, int height, int is_x) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >=1 && y < height -1) {
        float val = 0.0f;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <=1; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                float pixel = img[iy * width + ix];
                int kidx = (ky + 1) * 3 + (kx + 1);
                if (is_x != 0) {
                    val += pixel * sobel_x[kidx];
                } else {
                    val += pixel * sobel_y[kidx];
                }
            }
        }
        output[y * width + x] = val;
    }
}

__global__ void color_quantization(float *img, int width, int height, int levels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float step = 1.0f / levels;
        for (int c = 0; c < 3; c++) {
            float val = img[idx + c];
            val = floorf(val / step) * step;
            img[idx + c] = val;
        }
    }
}

__global__ void threshold(float *img, int size, float thresh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        img[idx] = img[idx] > thresh ? 1.0f : 0.0f;
    }
}

__global__ void dilate(float *img, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float max_val = img[idx];
        if (x > 0) max_val = fmaxf(max_val, img[idx - 1]);
        if (x < width - 1) max_val = fmaxf(max_val, img[idx + 1]);
        output[idx] = max_val;
    }
}

__global__ void gamma_correction(float *img, int size, float gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        img[idx] = powf(img[idx], gamma);
    }
}
"""

def cartoon_filter(img, edge_thresh=0.25, edge_dilate_iters=2, color_levels=5, gamma=0.8):
    ctx = create_cuda_context()  # Iniciar contexto CUDA

    try:
        # Compilar los kernels CUDA
        mod = SourceModule(kernel_code)
        sobel_filter = mod.get_function("sobel_filter")
        color_quantization = mod.get_function("color_quantization")
        threshold = mod.get_function("threshold")
        dilate = mod.get_function("dilate")
        gamma_correction = mod.get_function("gamma_correction")

        h, w, c = img.shape
        img_f = img.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Reservar memoria y copiar la imagen en escala de grises
        gray_gpu = cuda.mem_alloc(gray.nbytes)
        sobel_x_gpu = cuda.mem_alloc(gray.nbytes)
        sobel_y_gpu = cuda.mem_alloc(gray.nbytes)
        cuda.memcpy_htod(gray_gpu, gray)

        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1])

        # Aplicar filtros Sobel en X e Y
        sobel_filter(gray_gpu, sobel_x_gpu, np.int32(w), np.int32(h), np.int32(1), block=block, grid=grid)
        sobel_filter(gray_gpu, sobel_y_gpu, np.int32(w), np.int32(h), np.int32(0), block=block, grid=grid)

        sobel_x_res = np.empty_like(gray)
        sobel_y_res = np.empty_like(gray)
        cuda.memcpy_dtoh(sobel_x_res, sobel_x_gpu)
        cuda.memcpy_dtoh(sobel_y_res, sobel_y_gpu)

        grad_mag = np.sqrt(sobel_x_res**2 + sobel_y_res**2)
        grad_mag = np.clip(grad_mag, 0, 1)

        grad_mag_gpu = cuda.mem_alloc(grad_mag.nbytes)
        cuda.memcpy_htod(grad_mag_gpu, grad_mag)

        size = h * w
        threads = 256
        blocks = (size + threads - 1) // threads

        # Umbralización
        threshold(grad_mag_gpu, np.int32(size), np.float32(edge_thresh), block=(threads, 1, 1), grid=(blocks, 1))

        # Dilatación para engrosar bordes
        dilate_out = cuda.mem_alloc(grad_mag.nbytes)
        for _ in range(edge_dilate_iters):
            dilate(grad_mag_gpu, dilate_out, np.int32(w), np.int32(h), block=block, grid=grid)
            grad_mag_gpu, dilate_out = dilate_out, grad_mag_gpu

        edges = np.empty_like(grad_mag)
        cuda.memcpy_dtoh(edges, grad_mag_gpu)

        # Corrección gamma
        img_f_gpu = cuda.mem_alloc(img_f.nbytes)
        cuda.memcpy_htod(img_f_gpu, img_f)
        gamma_correction(img_f_gpu, np.int32(img_f.size), np.float32(gamma), block=(threads, 1, 1), grid=(blocks, 1))

        # Cuantización de colores
        color_quantization(img_f_gpu, np.int32(w), np.int32(h), np.int32(color_levels), block=block, grid=grid)

        # Inversión de gamma
        gamma_correction(img_f_gpu, np.int32(img_f.size), np.float32(1.0 / gamma), block=(threads, 1, 1), grid=(blocks, 1))

        img_quant = np.empty_like(img_f)
        cuda.memcpy_dtoh(img_quant, img_f_gpu)

        # Aplicar bordes como máscara negra
        edges_inv = 1 - edges
        edges_3c = np.repeat(edges_inv[:, :, np.newaxis], 3, axis=2)
        cartoon = img_quant * edges_3c
        cartoon = (cartoon * 255).astype(np.uint8)

        return cartoon

    finally:
        cleanup_cuda_context(ctx)  # Liberar el contexto CUDA
