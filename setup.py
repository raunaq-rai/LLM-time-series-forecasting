from setuptools import setup, find_packages

setup(
    name="m2_coursework",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "h5py",
        "torch",
        "transformers",
        "pytest",
        "scikit-learn"
    ],
)

