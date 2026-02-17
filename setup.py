#!/usr/bin/env python
"""Setup script for sequential-indicator-sim."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sequential-indicator-sim",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-performance 3D sequential indicator simulation with Gaussian variogram",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/timcjohnson/sequential-indicator-sim",
    packages=find_packages(),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    entry_points={
        "console_scripts": [
            "isim=sequential_indicator_sim:main",
        ],
    },
)
