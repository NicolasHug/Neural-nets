import setuptools
from os import path


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
install_requires = [x.strip() for x in all_reqs]

setuptools.setup(
    name="neuralnets",
    version="0.1.0",
    url="",

    author="Nicolas Hug",
    author_email="contact@nicolas-hug.com",

    description="A basic NN library",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=install_requires,

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
    ],
)
