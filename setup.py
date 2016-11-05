import setuptools
from setuptools import setup
from setuptools.command.install import install

setup(name='tankbuster',
      version='0.2.2',
      description='An image classifier trained to detect Soviet/Russian military vehicles',
      url='https://github.com/thiippal/tankbuster',
      author='Tuomo Hiippala',
      packages=['tankbuster'],
      package_dir={'tankbuster': 'tankbuster'},
      package_data={'tankbuster': ['*.py', 'engine/*.h5', 'cnn/*.py', 'engine/*.py']},
      author_email='tuomo.hiippala@iki.fi',
      license='MIT',
      keywords=['osint', 'imaging', 'classifier', 'deep learning', 'convolutional neural net'],
      download_url='https://github.com/thiippal/tankbuster/tarball/0.2.2',
      install_requires=["Pillow>=3.3.1",
                        "numpy>=1.11.1",
                        "Keras>=1.1.0",
                        "colorama>=0.3.7",
                        "h5py>=2.6.0"],
      classifiers=['Programming Language :: Python :: 2.7'])
