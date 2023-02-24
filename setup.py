from setuptools import setup

setup(
    name='gaussian-preact',
    version='0.1.0',
    author='Pierre Wolinski',
    author_email='pierre.wolinski@normalesup.org',
    packages=['gaussian_preact'],
    url='https://github.com/p-wol/gaussian-preact',
    license='LICENSE',
    description='Computations related to the paper "Gaussian Pre-Activations in Neural Networks: Myth or Reality?"',
    long_description=open('README.md').read(),
    install_requires=[
        "pytorch >= 1.13.1",
        "scipy",
        "numpy",
    ],
)
