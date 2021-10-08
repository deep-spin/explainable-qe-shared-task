# -*- coding: utf-8 -*-
from setuptools import find_packages, setup


setup(
    name="xlmr-qe",
    version="0.0.1",
    author="Ricardo Rei",
    author_email="ricardo.rei@unbabel.com",
    packages=find_packages(exclude=["tests"]),
    description="High-quality Machine Translation Quality Estimation using pre-trained Encoder-Decoder models",
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=[
        line.strip() for line in open("requirements.txt", "r").readlines()
    ]
)