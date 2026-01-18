from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "game_core",
        ["game_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="game_core",
    ext_modules=ext_modules,
)