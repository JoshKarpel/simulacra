from pathlib import Path

from setuptools import setup, find_packages

THIS_DIR = Path(__file__).parent

setup(
    name="simulacra",
    version="0.2.0",
    author="Josh Karpel",
    author_email="josh.karpel@gmail.com",
    description="A Python library for running simulations and generating visualizations.",
    long_description=Path("README.md").read_text(),
    url="https://github.com/JoshKarpel/simulacra",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: System :: Distributed Computing",
    ],
    packages=find_packages(exclude=["dev", "tests"]),
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
