import setuptools

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [#'tensorflow', - issues with my miniforge3 - apple silicone - tensorflow setup
                'tensorflow-probability',    
                'pandas',
                'scikit-learn',
                    ]
    
setuptools.setup(
    name='hivae2',
    version='0.12',
    url='https://github.com/gkoutos-group/hivae/',
    license='MIT',
    author='Andreas Karwath',
    author_email='a.karwath@bham.ac.uk',
    description="""HIVAE (Handling incomplete heterogeneous data using VAEs. - by Nazabal, et al., DOI: 10.1016/j.patcog.2020.107501, 2020)
Extenstion of implementations as easy to use Python library/tf2 version (a.karwath)
Further extension to procude imputation objects for sklearn API (a.karwath) """,
    packages=setuptools.find_packages(exclude=['examples']),
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    zip_safe=False
    )


