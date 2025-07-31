from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="microdiff-matdesign",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@example.com",
    description="Diffusion model framework for inverse material design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/microdiff-matdesign",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["torch[cuda]", "cupy"],
        "full": ["jupyter", "matplotlib", "plotly", "tqdm"],
        "dev": ["pytest", "black", "flake8", "mypy", "pre-commit"]
    },
    entry_points={
        "console_scripts": [
            "microdiff=microdiff_matdesign.cli:main",
        ],
    },
)