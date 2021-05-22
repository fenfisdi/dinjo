import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='DINJO',
    version='0.0.dev1',
    description='DINJO lets you find optimal values of initial value problems\' parameters',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Juan Esteban Aristizabal-Zuluaga',
    author_email='jeaz.git@gmail.com',
    license='gplv3',
    packages=['dinjo'],
    zip_safe=False
)
