from setuptools import setup

setup(name='nanoepiseg',
      version='0.1',
      description='Segmentation method for nanopore methylation log-likelihood ratios',
      url='https://github.com/snajder-r/nanoepiseg',
      author='Rene Snajder',
      license='MIT',
      packages=['nanoepiseg'],
      install_requires=[
          'numpy',
          # Due to scipy bug #11403 it doesn't work with scipy 1.5.
          # This constrained can be lifted if the bug is fixed
          'scipy==1.4.1',
          'pandas',
          'nanoepitools',
          'matplotlib'
      ],
      zip_safe=False)
