from setuptools import setup, find_packages

setup(
    name="rtvf-detection",
    version="0.1.0",
    description=(
        "Real-Time Vocal Fold Motion Tracking — "
        "replication of Koivu et al. (Laryngoscope, 2026)"
    ),
    author="Replication Study",
    python_requires=">=3.9,<3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow==2.10.0",
        "keras==2.10.0",
        "protobuf==3.19.6",
        "torch==1.12.1",      # upper-bounded by deeplabcut==2.3.6 (torch<=1.12)
        "torchvision==0.13.1",
        "opencv-python==4.8.1.78",
        "deeplabcut==2.3.6",
        "numpy==1.23.5",
        "pandas>=1.5,<2.0",
        "scipy>=1.9",
        "matplotlib>=3.6",
        "scikit-learn>=1.1",
        "Pillow>=9.0",
        "tqdm>=4.64",
        "PyYAML>=6.0",
        "pynvml>=11.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
