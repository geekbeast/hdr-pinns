from setuptools import setup, find_packages

def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read().strip()

version = read_file('VERSION')
readme = read_file('README.md')

setup(
    name='hdr-pinns',
    version=version,
    author='Matthew Tamayo-Rios',
    author_email='matthew@geekbeast.com',
    description='Pinn Foundation Models',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    keywords='machine-learning signature sequence time-series pinn deep learning',
    url='https://github.com/geekbeast/hdr-pinns',
    packages=find_packages(),
    install_requires=['torch>=2.5.0', 'numpy>=1.26.4', 'scikit-learn>=1.3.2', 'datasets', 'transformers', 'matplotlib', 'tokenizers', 'tqdm==4.66.1'],
    python_requires='>=3.11',
    classifiers=[
        'Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)