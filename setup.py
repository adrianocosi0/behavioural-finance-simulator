from setuptools import setup, find_packages

setup(
    name="behavioural_finance_simulations",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "streamlit",
    ],
    author="Adriano Cosi",
    description="Tools and notebooks for simulating portfolio strategies and analyzing my behavioral finance biases in investment decisions.",
)