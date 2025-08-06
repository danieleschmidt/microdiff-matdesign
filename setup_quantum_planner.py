"""Setup script for quantum task planner."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="quantum-task-planner",
    version="1.0.0",
    author="Terragon Labs",
    author_email="info@terragonlabs.com",
    description="Quantum-Inspired Task Planner: Advanced scheduling using quantum principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/quantum-inspired-task-planner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core requirements (lightweight)
        "dataclasses; python_version<'3.7'",
    ],
    extras_require={
        "full": [
            "numpy>=1.21.0",
            "matplotlib>=3.5.0", 
            "seaborn>=0.11.0",
            "networkx>=2.6.0",
            "psutil>=5.8.0",
            "scipy>=1.7.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0", 
            "networkx>=2.6.0",
        ],
        "performance": [
            "numpy>=1.21.0",
            "psutil>=5.8.0",
        ],
        "quantum": [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "quantum-planner=quantum_planner.main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/quantum-inspired-task-planner/issues",
        "Source": "https://github.com/danieleschmidt/quantum-inspired-task-planner",
        "Documentation": "https://quantum-task-planner.readthedocs.io/",
    },
    keywords="quantum scheduling task-planning optimization algorithms",
    include_package_data=True,
    package_data={
        "quantum_planner": ["configs/*.yaml", "templates/*.json"],
    },
)