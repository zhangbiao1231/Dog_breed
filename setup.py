from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dog-greed",
    version="1.1.0",
    packages=find_packages(exclude=['ext']),
    install_requires=[
        "torch>=1.3",
        "torchvision>=0.3",
        "opencv-python~=4.0",
        "yacs==0.1.6",
        "Vizer~=0.1.4",
    ],
    author="Zebulon Zhang",
    author_email="zhangbiao02@spic.com",
    description="High quality, fast, modular reference implementation of SSD in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangbiao1231/Dog_breed",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    python_requires=">=3.6",
    include_package_data=True,
)