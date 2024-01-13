from setuptools import find_packages, setup

setup(
    name="core",
    version="1.0",
    install_requires=[],
    packages=find_packages(exclude=["tools"]),
    extras_require={
        "all": ["numpy", "opencv-python", "tensorrt", "torch"]
    },
)
