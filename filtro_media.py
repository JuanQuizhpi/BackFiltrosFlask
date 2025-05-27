import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context
import time

cuda.init()

def filtro_media_cuda(img_array):
    mask_size = 31  

    ctx = make_default_context()
    try:
        height, width = img_array.shape
        output_array = np.empty_like(img_array)

        input_gpu = cuda.mem_alloc(img_array.nbytes)
        output_gpu = cuda.mem_alloc(img_array.nbytes)
        cuda.memcpy_htod(input_gpu, img_array)

        mod = SourceModule(f"""
        __global__ void filtro_media(unsigned char* input, unsigned char* output, int width, int height, int maskSize) {{
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int radius = maskSize / 2;
            int windowSize = maskSize * maskSize;

            if (x >= radius && x < (width - radius)) {{
                for (int y = radius; y < (height - radius); y++) {{
                    int sum = 0;
                    for (int dy = -radius; dy <= radius; ++dy) {{
                        for (int dx = -radius; dx <= radius; ++dx) {{
                            int px = x + dx;
                            int py = y + dy;
                            sum += input[py * width + px];
                        }}
                    }}
                    output[y * width + x] = sum / windowSize;
                }}
            }}
        }}
        """)

        filtro = mod.get_function("filtro_media")

        threads_per_block = 256
        blocks_per_grid = (width + threads_per_block - 1) // threads_per_block

        filtro(
            input_gpu, output_gpu,
            np.int32(width), np.int32(height), np.int32(mask_size),
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_grid, 1)
        )
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(output_array, output_gpu)

        return output_array

    finally:
        ctx.pop()
