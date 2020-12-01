import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cdslib_cmodel',
    version='0.0.0',
    description='Contagious Disease Simulation - Compartmental Models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Juan Esteban Aristizabal Zuluaga',
    author_email='jeaz.git@gmail.com',
    license='gplv3',
    packages=['cmodel'],
    zip_safe=False
)