from setuptools import find_packages, setup
from pathlib import Path

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / "requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="NEWS_CATEGRIZATION",
    version="0.1.0",
    author="Samarth Bhardwaj",
    description="News Category Classification using NLP and ML",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
)