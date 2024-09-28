from setuptools import setup, find_packages

setup(
    name='gradient_visualizer',
    version='0.1',
    description='Interactive Streamlit app to visualize gradient flow of a neural network',
    author='Cristian Leo',
    author_email='cristianleo120@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'streamlit',
        'plotly',
        'numpy',
    ],
)
