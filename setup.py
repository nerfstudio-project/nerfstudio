import os

from setuptools import setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

ext_modules = []
cmdclass = {}
setup_requires = []

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        setup_requires=setup_requires,
    )
