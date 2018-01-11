from setuptools import setup, find_packages

setup(
    name='KDRL',
    version='0.0.2',
    packages=find_packages(exclude=['tests']),
    author='obsproth',
    url='https://github.com/obsproth/KDRL',
    install_requires=['Keras>=2.0.0',
                      ],
)

