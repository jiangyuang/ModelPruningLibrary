import os
from setuptools import setup, find_packages
from torch.utils import cpp_extension

# Set up sparse conv2d extension
setup(name="sparse_conv2d",
      ext_modules=[cpp_extension.CppExtension("sparse_conv2d",
                                              [os.path.join("extension", "extension.cpp")],
                                              extra_compile_args=["-std=c++14", "-fopenmp"])],
      cmdclass={"build_ext": cpp_extension.BuildExtension})

# Set up mpl (model pruning library)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

DEPENDENCIES = ['torch', 'torchvision']

setup(name='mpl',
      version='0.0.1',
      description="Model Pruning Library",
      long_description=long_description,
      author="Yuang Jiang",
      author_email="yuang.jiang@yale.edu",
      url="https://github.com/jiangyuang/ModelPruningLibrary",
      packages=find_packages(),
      install_requires=DEPENDENCIES,
      )
