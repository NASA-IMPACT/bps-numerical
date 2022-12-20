from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

exec(open("code/bps_numerical/__version__.py").read())

setup(
    name="bps_numerical",
    version=__version__,
    description="Framework to analyze and rank genes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA-IMPACT/bps-numerical",
    author="NASA-IMPACT, Nish",
    author_email="np0069@uah.edu",
    # license="MIT",
    python_requires=">=3.7",
    package_dir={"bps_numerical": "code/bps_numerical"},
    packages=["bps_numerical", "bps_numerical.classification", "bps_numerical.misc"],
    install_requires=required,
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Gene Expression",
        "Topic :: Machine Learning",
        "Topic :: Feature Importance",
        "Topic :: Feature Ranking",
    ],
    zip_safe=False,
)
