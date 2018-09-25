# Installation method
- We will explain the installation method.
- Because RDKit is used, Anaconda is required. if you do not want to pollute your environment, we recommend using Docker.
    - Please see [Using the Docker environment](https://github.com/KCF-Convoy/kcfconvoy/wiki/Using-the-Docker-environment)

## Requirements
- Python (3.4 -)
    - RDKit
    - Cython
    - matplotlib
    - networkx
    - numpy
    - pandas
    - scipy

## In the case of using conda
- Please install Anaconda with reference below:
    - [Anaconda Documentation-Installation](https://docs.anaconda.com/anaconda/install/)
    - In our test we used miniconda.
- After installing Anaconda

```
$ git pull git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy
$ conda env create --name kcfconvoy --file conda-env.yml
$ conda activate kcfconvoy
$ pip install kcfconvoy
(kcfconvoy) $ python
>> import kcfconvoy
```

## In the case of NOT using conda
- Please install RDKit
    - [RDKit Python Documentation](https://www.rdkit.org/docs/GettingStartedInPython.html)
    - It is very difficult.
- After installing RDKit

### Using PyPI
```
$ pip install kcfconvoy Cython matplotlib networkx numpy pandas scipy
$ python
>> import kcfconvoy
```

### Using GitHub Source
```
$ git pull git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy
$ python setup.py install
```

### Using Wheel Package
```
$ wget (TODO wheel url)
$ pip install (TODO wheel file name)
```

## Test
- Test to check if KCF-Convoy was installed correctly

```
$ git pull git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy/tests
$ python -m unittest discover
```
