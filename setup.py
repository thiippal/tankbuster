import setuptools
from setuptools import setup
from setuptools.command.install import install

setup(name='tankbuster',
      version='0.1.3',
      description='An image classifier trained to detect Soviet/Russian T-72 tanks',
      url='https://github.com/thiippal/tankbuster',
      author='Tuomo Hiippala',
      packages=['tankbuster'],
      package_dir={'tankbuster': 'tankbuster'},
      package_data={'tankbuster': ['*.py', 'classifier/*.json', 'classifier/*.hdf5', 'cnn/*.py', 'engine/*.py']},
      author_email='tuomo.hiippala@iki.fi',
      license='MIT',
      keywords=['osint', 'imaging', 'classifier'],
      download_url='https://github.com/thiippal/tankbuster/tarball/0.1.3',
      install_requires=["Pillow>=3.3.1",
                        "numpy>=1.11.1",
                        "Keras>=1.0.8",
                        "colorama>=0.3.7",
                        "h5py>=2.6.0"],
      classifiers=['Programming Language :: Python :: 2.7'])
