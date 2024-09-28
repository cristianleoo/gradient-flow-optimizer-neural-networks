from setuptools import setup, find_packages

setup(
    name='gradient_visualizer',
    version='0.1',
    description='Interactive Streamlit app to visualize gradient flow of a neural network',
    author='Your Name',
    author_email='youremail@example.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'streamlit',
        'matplotlib',
        'numpy',
    ],
)
