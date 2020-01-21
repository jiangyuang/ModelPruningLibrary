from distutils.core import setup

DEPENDENCIES = ['torch', 'torchvision']

setup(name='Model Pruning Libarary',
      version='1.0',
      description='',
      author='Yuang Jiang',
      author_email='yuang.jiang@yale.edu',
      url='https://github.com/jiangyuang/ModelPruningLibrary',
      packages=['bases', 'bases/nn', 'configs', 'datasets', 'utils'],
      install_requires=DEPENDENCIES,
     )