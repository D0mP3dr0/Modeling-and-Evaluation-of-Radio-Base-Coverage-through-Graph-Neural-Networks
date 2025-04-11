#!/usr/bin/env python3
"""
Setup script for the RBS Analysis project.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Get the long description from the README file
here = Path(__file__).resolve().parent
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Load version from the __init__.py file
with open(os.path.join(here, 'src', '__init__.py'), encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.1.0'  # Default version if not found

# Read requirements from requirements.txt
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='rbs_analysis',
    version=version,
    description='A tool for analyzing radio base station data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/rbs_analysis',
    author='Your Name',
    author_email='your.email@example.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Telecommunications Industry',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='radio, base station, analysis, telecom, gis',
    packages=find_packages(),
    python_requires='>=3.8, <4',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'rbs-analysis=rbs_analysis:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/rbs_analysis/issues',
        'Source': 'https://github.com/yourusername/rbs_analysis',
    },
) 