"""Standalone build for the malsim_native C++ extension.

Mirrors how the Rust/PyO3 sibling implementation (rust/, built with
`maturin develop`) is built: a self-contained subdirectory with its own
build config, built directly into the active venv, kept out of the main
package's setuptools build so the two native backends don't collide.

Usage (into the currently active venv):
    uv pip install ./cpp
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        'malsim_native',
        ['src/attack_graph_index.cpp'],
        cxx_std=17,
    ),
]

setup(
    name='malsim-native',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
