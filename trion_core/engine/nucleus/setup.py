from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="trion_nucleus",
    ext_modules=[
        CUDAExtension(
            name="trion_nucleus",
            sources=[
                os.path.join(this_dir, "src", "nucleus_bind.cpp"),
                os.path.join(this_dir, "src", "nucleus_sample.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    '-allow-unsupported-compiler',
                    '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' 
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
