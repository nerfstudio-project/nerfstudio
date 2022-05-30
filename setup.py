import os
from setuptools import setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True

ext_modules = []
cmdclass = {}
setup_requires = []
if USE_CUDA:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        ext_modules = [
            CUDAExtension(
                name="radiance_cuda",
                sources=[
                    "radiance/cuda/radiance_cuda.cpp",
                    "radiance/cuda/radiance_cuda_kernel.cu",
                ],
                include_dirs=[os.path.join("radiance", "cuda", "include")],
                extra_compile_args={
                    "cxx": [],
                    "nvcc": [],
                },
                verbose=True,
            ),
        ]
        cmdclass = {"build_ext": BuildExtension}
        setup_requires = ["pybind11>=2.5.0"]
    except:
        print("Failed to build the CUDA extension.")

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        setup_requires=setup_requires,
    )
