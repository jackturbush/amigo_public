import os
import sys
from setuptools import setup, Extension
from subprocess import check_output
import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

inc_dirs, lib_dirs, libs = [], [], []

inc_dirs.append("include")
headers = glob.glob("include/*.h")

# Get the include for pybind11
pybind11_include = pybind11.get_include()

# Construct the A2D path from a guess...
a2d_include = os.path.join(os.environ.get("HOME"), "git", "a2d", "include")

# metis_include = os.path.join(
#     os.environ.get("HOME"), "git", "tacs", "extern", "metis", "include"
# )
# lib_dirs.append(
#     os.path.join(os.environ.get("HOME"), "git", "tacs", "extern", "metis", "lib")
# )
# libs.append("metis")

# Optionally write the include path to a header or config
with open("include/amigo_include_paths.h", "w") as f:
    f.write(f"#ifndef AMIGO_INCLUDE_PATHS_H\n")
    f.write(f"#define AMIGO_INCLUDE_PATHS_H\n")
    f.write(f'#define A2D_INCLUDE_PATH "{a2d_include}"\n')
    f.write(f'#define AMIGO_INCLUDE_PATH "{os.path.abspath("include")}"\n')
    f.write(f"#endif  // AMIGO_INCLUDE_PATHS_H\n")


if sys.platform == "darwin":
    from distutils import sysconfig

    vars = sysconfig.get_config_vars()
    vars["LDSHARED"] = vars["LDSHARED"].replace("-bundle", "-dynamiclib")

# Create the Extension
ext_modules = [
    Extension(
        "amigo.amigo",
        sources=["amigo/amigo.cpp"],
        depends=headers,
        include_dirs=inc_dirs,
        libraries=libs,
        library_dirs=lib_dirs,
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="amigo",
    ext_modules=ext_modules,
    include_dirs=[pybind11_include, a2d_include],
)
