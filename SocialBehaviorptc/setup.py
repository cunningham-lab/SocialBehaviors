from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='vipoint',
      version='0.0.1',
      description='SocialBehaviorModelling',
      author='Luhuan Wu',
      packages=['ssm_ptc', 'ssm_ptc.message_passing', 'ssm_ptc.models',
                'ssm_ptc.observations', 'ssm_ptc.transformations',
                'ssm_ptc.transitions', 'ssm_ptc.distributions'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[np.get_include(),],
      )
