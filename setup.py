from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cipher_cpp",  # Name of the module
        ["cipher_cpp/cipher_pybind.cpp", "cipher_cpp/cipher.cpp"],  # Source files
        include_dirs=["cipher_cpp"],  # Include directories
        extra_compile_args=[
            '-std=c++17',     # Use C++17 standard
            '-fopenmp',       # Enable OpenMP for parallelism
            '-O3',            # Enable high optimization level
            '-march=native',  # Enable architecture-specific optimizations
            '-ffast-math',    # Allow aggressive floating-point optimizations
            '-ftree-vectorize',  # Encourage the compiler to vectorize loops
            '-funroll-loops'
        ],
        extra_link_args=['-fopenmp']  # Link with OpenMP
    )
]

# Setup function
setup(
    name="cipher",
    version="0.0",
    author="Jinwon Pyo",
    description="Cipher library",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "pybind11>=2.12",    # Ensure pybind11 is installed
        "cupy>=13.1.0",        # Specify the version of sympy you need
        "numpy>=1.26.3",        # Specify the version of sympy you need
        "singleton-timer>=0.0",
        "nvtx>=0.2.10"
    ],
    zip_safe=False,
)
