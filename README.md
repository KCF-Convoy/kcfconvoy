# KCF-Convoy: efficient Python package to convert KCF chemical substructure fingerprints

KCF-Convoy is a new Python package to generate KCF formats and KCF-S fingerprints from Molfile, SDF, SMILES and InChI seamlessly.
Obtained KCF-S were applied to a series of machine learning binary classification methods to distinguish herbicides from other pes- ticides, and also to find characteristic substructures in a specific genus.

Visit [GitHub Wiki](https://github.com/KCF-Convoy/kcfconvoy/wiki) for more details.

## Usage

Use conda

```bash
$ git clone git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy
$ conda install -c conda-forge rdkit
$ python3 setup.py install
```

Use docker and docker-compose

```bash
$ git clone git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy
$ docker-compose up -d --build
$ docker-compose exec app bash
root@e9ea26cc0217:/opt/kcfconvoy# python3
Python 3.8.0 | packaged by conda-forge | (default, Nov 22 2019, 19:11:38)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import kcfconboy
>>>
```

### Run tests

```bash
$ cd tests
$ python -m unittest *.py
```

## Contact

- Author: kotera@preferred.jp
- Maintainer: suecharo@g.ecc.u-tokyo.ac.jp

## License

MIT

## Cite

- Masayuki Sato, Hirotaka Suetake, Masaaki Kotera (2018) "KCF-Convoy: efficient Python package to convert KEGG Chemical Function and Substructure fingerprints", bioRxiv, doi: https://doi.org/10.1101/452383 

- Kotera, Masaaki, et al. "KCF-S: KEGG Chemical Function and Substructure for improved interpretability and prediction in chemical bioinformatics." BMC systems biology 7.6 (2013): 1-17. doi: https://doi.org/10.1186/1752-0509-7-S6-S2
