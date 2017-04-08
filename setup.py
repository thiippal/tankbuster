import setuptools
from setuptools import setup
from setuptools.command.install import install

setup(name='tankbuster',
      version='0.3.1',
      description='A neural network trained to detect Soviet/Russian military vehicles in photographs',
      url='https://github.com/thiippal/tankbuster',
      author='Tuomo Hiippala',
      packages=['tankbuster'],
      package_dir={'tankbuster': 'tankbuster'},
      package_data={'tankbuster': ['*.py', 'cnn/*.py', 'engine/*.py']},
      author_email='tuomo.hiippala@iki.fi',
      license='MIT',
      keywords=['osint', 'computer vision', 'object recognition', 'deep learning', 'neural network'],
      download_url='https://github.com/thiippal/tankbuster/archive/0.3.1.tar.gz',
      install_requires=["Keras>=2.0.2",
                        "tensorflow>=1.0.1",
                        "h5py>=2.7.0"],
      classifiers=['Programming Language :: Python :: 2.7'])
