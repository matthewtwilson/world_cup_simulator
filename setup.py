from setuptools import setup

setup(
    name="world_cup_simulator",
    version="0.1.0",
    py_modules=["world_cup_simulator"],
    python_requires=">=3.11",
    install_requires=[
        "pandas",
        "numpy",
    ],
    description="A World Cup 2026 simulation engine.",
    author="",
    author_email="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
