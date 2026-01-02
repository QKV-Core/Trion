from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# src klasörünün içindeki dosyalara işaret ediyoruz
sources = [
    os.path.join('src', 'attention_fused.cu'),
    os.path.join('src', 'attention_bindings.cpp'),
]

setup(
    name='attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='attention_cuda',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '--use_fast_math', 
                    '-allow-unsupported-compiler',
                    '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' 
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)