import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='wetlabtools',
    version='0.0.1',
    author='Nicolas Goldbach',
    author_email='nicolas.goldbach@epfl.ch',
    description='Software tools for wet lab tasks',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)