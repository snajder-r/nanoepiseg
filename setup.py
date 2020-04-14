from setuptools import setup

setup(name='nanoepiseg',
      version='0.1',
      description='Segmentation method for nanopore methylation log-likelihood ratios',
      url='https://github.com/snajder-r/nanoepiseg',
      author='Rene Snajder',
      license='MIT',
      packages=['nanoepiseg'],
      install_requires= [
          'numpy',
          'scipy'
      ],
      zip_safe=False)
