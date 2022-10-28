import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='topicflow',
    version='0.0.1',
    packages=['topicflow'],
    description='Generative text modelling based on tensorflow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='Jannis Zeller',
    author_email='zeller@physik.rwth-aachen.de',
    keywords=['zeller', 'tensorflow', 'modelling', 'NLP'],
    license='GNU v3',
    install_requires=['matplotlib', 'numpy', 'tensorflow', 'tqdm', 
                      'tensorflow_probability'],
)