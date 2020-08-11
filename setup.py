from setuptools import setup, find_packages

setup(
    name='vae',
    author='Avan Suinesiaputra',
    author_email='avan.sp@gmail.com',
    version='0.1.0',
    license='LICENSE',
    description='Variational Auto Encoders',
    packages=find_packages(),
    long_description=open('README.md').read(),
)