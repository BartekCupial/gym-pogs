import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    # Information
    name="gym-pogs",
    description="Partially Observable Graph Search (POGS) environment for OpenAI Gym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.0",
    url="https://github.com/BartekCupial/gym-pogs/",
    author="Bartłomiej Cupiał",
    license="MIT",
    keywords="reinforcement learning ai",
    install_requires=[
        "gymnasium ~= 0.29",
        "networkx",
        "numpy",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "black",
            "isort>=5.12",
            "pytest<8.0",
            "flake8",
            "pre-commit",
            "twine",
        ]
    },
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./", include=["gym_pogs*"]),
    include_package_data=True,
    python_requires=">=3.8",
)
