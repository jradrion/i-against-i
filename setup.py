from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='i-against-i',
      version='0.1',
      description='',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/jradrion/i-against-i/',
      author='Jeffrey Adrion',
      author_email='jeffreyadrion@gmail.com',
      license='MIT',
      packages=find_packages(exclude=[]),
      install_requires=[
          "msprime>=0.7.4",
          "scikit-learn>=0.22.1",
          "matplotlib>=3.1.3",
          "scikit-allel>=1.2.1"],
      scripts=[
            "iai/iai-simulate",
            "iai/iai-train",
            "iai/iai-predict"],
      zip_safe=False,
      setup_requires=[],
)

