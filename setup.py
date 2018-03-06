from setuptools import setup
setup(
    name='multiregression',
    description='Python 3 module for doing simultaneous linear regression', 
    version='1.0.0',
    packages=['multiregression'],
    author="Laurent Fasnacht",
    author_email="l@libres.ch", 
    url = 'https://github.com/UniNE-CHYN/multiregression',
    install_requires=[
        'numpy',
    ],    
)
