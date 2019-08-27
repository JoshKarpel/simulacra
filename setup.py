import re
from pathlib import Path

from setuptools import setup, find_packages

THIS_DIR = Path(__file__).parent


def find_version():
    """Grab the version out of simulacra/version.py without importing it."""
    version_file_text = (THIS_DIR / "simulacra" / "version.py").read_text()
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file_text, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="simulacra",
    version=find_version(),
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
        "Programming Language :: Python :: 3.7",
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
