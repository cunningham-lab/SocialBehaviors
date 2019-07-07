from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='vipoint',
      version='0.0.1',
      description='SocialBehaviorModelling',
      author='Luhuan Wu',
      packages=['socialbehavior', 'socialbehavior.message_passing', 'socialbehavior.models',
                'socialbehavior.observations', 'socialbehavior.transformations'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      )
