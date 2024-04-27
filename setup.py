from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cipher_cpp",                               # Name of the module
        # Correctly list source files
        ["cipher_cpp/cipher_pybind.cpp", "cipher_cpp/cipher.cpp"],
        # Update include directory if needed
        include_dirs=["cipher_cpp"],
        # Additional flags (like C++ standard)
        extra_compile_args=['-std=c++17'],
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
