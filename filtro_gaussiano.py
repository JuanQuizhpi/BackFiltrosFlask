import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from time import time

cuda.init()

def generate_gaussian_kernel(size, sigma):
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    normalization_factor = 1 / (2 * np.pi * sigma**2)

    total = 0.0
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            exponent = -(dx**2 + dy**2) / (2 * sigma**2)
            value = normalization_factor * np.exp(exponent)
            kernel[y, x] = value
            total += value

    # Normalizamos para que la suma del kernel sea 1
    kernel /= total
    return kernel

def filtro_gaussiano_cuda(image):
    mask_size = 41  # Tamaño fijo de la máscara 

    device = cuda.Device(0)
    context = device.make_context()

    try:
        sigma = 0.3 * ((mask_size - 1) * 0.5 - 1) + 0.8
        kernel = generate_gaussian_kernel(mask_size, sigma)

        height, width = image.shape
        kCenterX = mask_size // 2
        kCenterY = mask_size // 2

        gauss_kernel_code = SourceModule("""
        __global__ void filtroGaussianoCUDA(
            const unsigned char* imgEntrada,
            unsigned char* imgSalida,
            int ancho, int alto,
            const float* matrizKernel,
            int tamKernelX, int tamKernelY,
            int centroX, int centroY) {

            int coordX = blockIdx.x * blockDim.x + threadIdx.x;
            int coordY = blockIdx.y * blockDim.y + threadIdx.y;

            if (coordX >= ancho || coordY >= alto) return;

            float acumulado = 0.0f;

            for (int fila = 0; fila < tamKernelY; ++fila) {
                for (int col = 0; col < tamKernelX; ++col) {
                    int posX = coordX + col - centroX;
                    int posY = coordY + fila - centroY;

                    bool dentro = (posX >= 0 && posX < ancho && posY >= 0 && posY < alto);
                    if (dentro) {
                        int idxImagen = posY * ancho + posX;
                        int idxKernel = fila * tamKernelX + col;

                        float intensidad = static_cast<float>(imgEntrada[idxImagen]);
                        float peso = matrizKernel[idxKernel];

                        acumulado += intensidad * peso;
                    }
                }
            }

            int resultado = __float2int_rn(acumulado);
            resultado = max(0, min(255, resultado));
            imgSalida[coordY * ancho + coordX] = static_cast<unsigned char>(resultado);
        }
        """)

        input_flat = image.astype(np.uint8).flatten()
        output_flat = np.empty_like(input_flat)
        kernel_flat = kernel.astype(np.float32).flatten()

        d_input = cuda.mem_alloc(input_flat.nbytes)
        d_output = cuda.mem_alloc(output_flat.nbytes)
        d_kernel = cuda.mem_alloc(kernel_flat.nbytes)

        cuda.memcpy_htod(d_input, input_flat)
        cuda.memcpy_htod(d_kernel, kernel_flat)

        block_size = (16, 16, 1)
        grid_size = ((width + 15) // 16, (height + 15) // 16)

        kernel_func = gauss_kernel_code.get_function("filtroGaussianoCUDA")
        kernel_func(d_input, d_output,
                    np.int32(width), np.int32(height),
                    d_kernel,
                    np.int32(mask_size), np.int32(mask_size),
                    np.int32(kCenterX), np.int32(kCenterY),
                    block=block_size, grid=grid_size)

        cuda.memcpy_dtoh(output_flat, d_output)

        result_image = output_flat.reshape((height, width))

        return result_image

    finally:
        context.pop()
