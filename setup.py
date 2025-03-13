from setuptools import find_packages, setup
from typing import List
import os

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Reads the requirements file and returns a list of dependencies."""
    requirements = []
    
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found. No dependencies will be installed.")
        return requirements
    
    with open(file_path, "r", encoding="utf-8") as file_obj:
        requirements = [req.strip() for req in file_obj.readlines()]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)  # Remove editable install flag
    
    return requirements

# Read README.md with UTF-8 encoding
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="loanguard",
    version="0.0.1",
    author="Shrey",
    author_email="srkkadia@gmail.com",
    description="A loan risk prediction system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8.20",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
