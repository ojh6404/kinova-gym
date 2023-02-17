import os

from setuptools import find_packages, setup

with open(os.path.join("kinova_gym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="kinova_gym",
    description="Set of kinova robot environments based on PyBullet physics engine and gym.",
    author="Jihoon Oh",
    author_email="ojh6404@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ojh6404/kinova_gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={"kinova_gym": ["version.txt"]},
    version=__version__,
    install_requires=["gym~=0.26", "pybullet", "numpy", "scipy"],
    extras_require={
        "develop": ["pytest-cov", "black", "isort", "pytype", "sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
