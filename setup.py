import setuptools
from setuptools import setup
from setuptools.command.install import install

# TODO Check http://peterdowns.com/posts/first-time-with-pypi.html

setup(name='tankbuster',
      version='0.0.1',
      description='An image classifier trained to detect Soviet/Russian tanks',
      url='https://github.com/thiippal/tankbuster',
      author='Tuomo Hiippala',
      packages=['buster'],
      package_dir={'buster': 'buster'},
      package_data={'buster': ['*.py', 'classifier/*.json', 'classifier/*.hdf5', 'cnn/*.py', 'engine/*.py']},
      author_email='tuomo.hiippala@iki.fi',
      license='MIT',
      keywords=['osint', 'imaging', 'classifier'],
      download_url='https://github.com/thiippal/tankbuster/tarball/0.0.1',  # Still missing
      install_requires=["Pillow",
                        "numpy",
                        "keras"],
      classifiers=['Programming Language :: Python :: 2.7'])
