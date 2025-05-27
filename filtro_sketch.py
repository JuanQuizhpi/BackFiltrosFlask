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

# Kernel CUDA 
kernel_code = """
__global__ void invert_image(float *img, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f - img[idx];
    }
}

__global__ void dodge_blend(float *gray, float *blurred, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float denom = 1.0f - blurred[idx];
        denom = denom < 0.01f ? 0.01f : denom;
        output[idx] = gray[idx] / denom;
        output[idx] = output[idx] > 1.0f ? 1.0f : output[idx];
    }
}
"""

def sketch_filter(img, blur_kernel=21):
    ctx = create_cuda_context()  # Inicializar contexto
    try:
        # Compilar el kernel CUDA
        mod = SourceModule(kernel_code)
        invert_image = mod.get_function("invert_image")
        dodge_blend = mod.get_function("dodge_blend")

        h, w = img.shape[:2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        img_flat = img_gray.flatten()
        size = h * w

        threads = 256
        blocks = (size + threads - 1) // threads

        #Inversión de imagen
        img_gpu = cuda.mem_alloc(img_flat.nbytes)
        inv_gpu = cuda.mem_alloc(img_flat.nbytes)
        cuda.memcpy_htod(img_gpu, img_flat)
        invert_image(img_gpu, inv_gpu, np.int32(size), block=(threads, 1, 1), grid=(blocks, 1))

        # Descargar imagen invertida para aplicar desenfoque
        img_inv = np.empty_like(img_flat)
        cuda.memcpy_dtoh(img_inv, inv_gpu)
        img_inv_2d = img_inv.reshape((h, w))
        img_blur = cv2.GaussianBlur(img_inv_2d, (blur_kernel, blur_kernel), 0)

        # --- Dodge blending ---
        img_blur_flat = img_blur.flatten().astype(np.float32)
        blur_gpu = cuda.mem_alloc(img_blur_flat.nbytes)
        out_gpu = cuda.mem_alloc(img_flat.nbytes)
        cuda.memcpy_htod(blur_gpu, img_blur_flat)
        dodge_blend(img_gpu, blur_gpu, out_gpu, np.int32(size), block=(threads, 1, 1), grid=(blocks, 1))

        # Descargar resultado final
        result_flat = np.empty_like(img_flat)
        cuda.memcpy_dtoh(result_flat, out_gpu)
        result_img = (result_flat.reshape((h, w)) * 255).astype(np.uint8)

        return result_img

    finally:
        cleanup_cuda_context(ctx)  # Liberar contexto CUDA
