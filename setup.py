# coding: utf-8
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


def main():
    setup(
        name="kcfconvoy",
        version="0.0.4",
        url="https://github.com/KCF-Convoy",
        author="maskot1977",
        author_email="maskot1977@gmail.com",
        description="KCF convoy: requires rdkit installed in advance",
        license="MIT",
        keywords="bio-Informatics kcf kcfconvoy smiles",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: MIT License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        packages=find_packages(),
        install_requires=requirements,
        zip_safe=False
    )


if __name__ == "__main__":
    main()
