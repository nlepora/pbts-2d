"""Setup file for "Pose Based Tactile Servoing for 2D Surfaces and Edges"
"""

from setuptools import setup

base_deps = ["numpy","pandas","matplotlib"]
deep_learning_deps = ["tensorflow==2.3.0","scikit-learn"]
deep_learning_gpu_deps = ["tensorflow-gpu==2.3.0","scikit-learn"]
optimization_deps = ["hyperopt"]

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="pbts-2d",
    version="1.0",
    description="Pose Based Tactile Servoing for 2D Surfaces and Edges",
    license="GPLv3",
    long_description=long_description,
    author="Nathan Lepora",
    author_email="n.lepora@bristol.ac.uk",
    url="https://github.com/nlepora/pbts-2d/",
    packages=["pbts_2d"],
    install_requires=[base_deps],
    extras_require={
        "optimization": optimization_deps,
        "deep_learning": deep_learning_deps,
        "deep_learning_gpu": deep_learning_gpu_deps,
    },
)
