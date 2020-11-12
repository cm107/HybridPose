from setuptools import setup, find_packages
import HybridPose

packages = find_packages(
        where='.',
        include=['HybridPose*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='HybridPose',
    version=HybridPose.__version__,
    description='HybridPose fork turned into package.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cm107/HybridPose",
    author='Clayton Mork',
    author_email='mork.clayton3@gmail.com',
    license='MIT License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires='>=3.7'
)