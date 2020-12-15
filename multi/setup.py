import setuptools

with open("README.md", "r") as rm:
    long_description = rm.read()

setuptools.setup(
    name="multi",
    version="0.1",
    author="Philipp Diercks",
    author_email="philipp.diercks@bam.de",
    description="A tiny package to handle operations that come up repeatedly in our multiscale method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=["fenics_helpers @ https://github.com/BAMresearch/fenics_helpers/tarball/use_find_packages"],
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
    ],
)
