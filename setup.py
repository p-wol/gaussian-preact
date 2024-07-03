from setuptools import setup, find_packages

setup(
    name='gaussian-preact',
    version='0.2.0',
    author='Pierre Wolinski',
    author_email='pierre.wolinski@normalesup.org',
    packages=find_packages('src'),
    package_dir={"": 'src'},
    url='https://github.com/p-wol/gaussian-preact',
    license='LICENSE',
    description='Computations related to the paper "Gaussian Pre-Activations in Neural Networks: Myth or Reality?"',
    long_description=open('README.md').read(),
    install_requires=[
        "scipy >= 1.9.3",
        "numpy >= 1.23.5",
        "matplotlib >= 3.8.0"
    ],
)
