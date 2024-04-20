from struct import pack
from setuptools import setup, find_packages
setup(
    name='smoothquant',
    version='9.9.9',
    packages=find_packages(exclude=['figures', 'act_scales'])
)
