from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(name="pyforce-rl",
      version="0.0.4",
      author="Ole Meyer",
      author_email="dev@olemeyer.com",
      description=("PyForce - A simple reinforcement learning library"),
      install_requires=requirements,
      packages=find_packages(exclude=["examples","dist","build","evals"]),
      long_description=long_description,
      long_description_content_type='text/markdown'
      
      )
