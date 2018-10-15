# Example usage of KCF-convoy for machine learning

- Importing existing libraries. The sklearn.ensemble library may show a future warning.

```python
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

# Libraries for machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
```

    /opt/conda/envs/kcfconvoy/lib/python3.6/site-packages/sklearn/ensemble/weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d

- Inporting libraries defined in kcfconvoy.

```python
import kcfconvoy as kcf # KCF Convoy library
from kcfconvoy import Classifiers # a set of classifiers for machine learning
```

## Input compound structures in BRITE

- Input compound structures using KEGG BRITE https://www.genome.jp/kegg/brite.html

Example file can be obtained by clicking the <b>[Download htext]</b> link in https://www.genome.jp/kegg-bin/get_htext?ko01002.keg . KCF Convoy provides a method `.input_from_brite()` to input compounds from KEGG BRITE. RDKit may generate many warnings, which are because of the chemical structures defined in KEGG.

```python
# Please download "br08007.keg" file by clicking "Download htext" link from https://www.genome.jp/kegg-bin/get_htext?br08007.keg
# and put it in the appropriate directory.
!date
brite = './kegg/br08007.keg'
kcfmat = kcf.KCFmat()
kcfmat.input_from_brite(brite)
!date
```

    Mon Oct 15 07:23:29 UTC 2018


    RDKit WARNING: [07:23:30]  S group MUL ignored on line 75
    RDKit WARNING: [07:23:33] WARNING: Omitted undefined stereo
    RDKit WARNING: [07:23:35] WARNING: Omitted undefined stereo
    RDKit WARNING: [07:23:38] WARNING: Omitted undefined stereo
    RDKit WARNING: [07:23:39] WARNING: Omitted undefined stereo
    RDKit WARNING: [07:23:40] WARNING: Charges were rearranged

    RDKit ERROR: [07:33:55] Explicit valence for atom # 0 Si, 8, is greater than permitted


    Mon Oct 15 07:34:36 UTC 2018

- Check the number of compounds by `.cpds`.
- Note that we only made the collection of independent KCF vectors at this moment. The KCF matrix will be calculated later.

```python
len(kcfmat.cpds)
```

    918

- Check the grouping of compounds defined in KEGG BRITE by `.brite_group`

```python
kcfmat.brite_group
```

    {'A:<b>Pesticides</b>': ['',
      '',
      '',
      'C18498',
      'C18723',
      'C18568', ...
     'C:Diphenyl ether herbicides': ['C19065',
      'C19066',
      'C11063',
      'C11064',
      'C11065'],
     'C:Carbamate insecticide': ['C19130',
      'C18906',
      'C11071',
      'C18948',
      'C18949',
      'C18950',
      'C18951',
      'C18953',
      'C18954',
      'C18955',
      'C18956'],
     'C:Fumigants': ['C08880', 'C01998', 'C19033', 'C19034', 'C00829'],
     'C:Synergists': ['C19144', 'C19147', 'C19145', 'C19146']}

- Check the name of the compound groups defined in KEGG BRITE by `.list_brite_groups()`.

```python
kcfmat.list_brite_groups()
```

    [(0, 'A:<b>Obsolete pesticides</b>', 313),
     (1, 'A:<b>Pesticides</b>', 741),
     (2, 'B:Fungicides', 240),
     (3, 'B:Herbicides', 365),
     (4, 'B:Insect growth regulators', 30),
     (5, 'B:Insecticides', 299),
     (6, 'B:Others', 33),
     (7, 'B:Plant growth regulators', 51),
     (8, 'B:Rodenticides', 35),
     (9, 'C:Acaricides', 44),
     (11, 'C:Amide fungicides', 39),
     (12, 'C:Amide herbicides', 25),
     (13, 'C:Anilide herbicides', 26),
     (14, 'C:Antibiotic fungicides', 10),
     (18, 'C:Auxins', 11),
     (19, 'C:Benzoic acid herbicides', 15),
     (22, 'C:Carbamate fungicides', 12),
     (23, 'C:Carbamate insecticide', 11),
     (24, 'C:Carbamate insecticides', 25),
     (25, 'C:Carbanilate herbicides', 11),
     (26, 'C:Chitin synthesis inhibitors', 12),
     (27, 'C:Conazole fungicides', 31),
     (29, 'C:Coumarin rodenticides', 11),
     (32, 'C:Dicarboximide fungicides', 13),
     (33, 'C:Dinitroaniline herbicides', 13),
     (36, 'C:Dithiocarbamate fungicides', 14),
     (39, 'C:Growth inhibitors / retardants', 15),
     (45, 'C:Insect repellents', 11),
     (57, 'C:Organochlorine insecticides', 20),
     (59, 'C:Organophosphorus fungicides', 10),
     (60, 'C:Organophosphorus herbicides', 11),
     (61, 'C:Organophosphorus insecticides', 108),
     (63, 'C:Others', 185),
     (64, 'C:Phenoxy herbicides', 23),
     (65, 'C:Phenylurea herbicides', 27),
     (69, 'C:Pyrethroid insecticides', 28),
     (72, 'C:Pyrimidine fungicides', 12),
     (75, 'C:Sulfonylurea herbicides', 26),
     (79, 'C:Thiocarbamate herbicides', 17),
     (80, 'C:Triazine herbicides', 24),
     (83, 'C:Urea herbicides', 13),
     (84, 'D:', 1051)]

## Classifying compounds into two groups

- The users can classify compounds into two groups by specifiying the class shown above. KCF-convoy provides a method `.brite_class()` to give a binary classification describing whether or not a group belongs to the specified group. In the case provided below, `(3, 'B:Herbicides', 365)` is specified, and the user will deal with the classification of herbicides from others.

```python
classes = kcfmat.brite_class(3)
```

- Check the classification results, which will be used as a training data

```python
print(classes)
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

## Supervised binary classification using existing chemical fingerprints

- KCF Convoy provides `Classifiers` for a number of classifiers, and `.calc_fingerprints()` for a set of existing chemical fingerprints (defined in RDKit).

```python
fingerprints = ["PatternFingerprint", "LayeredFingerprint", "RDKFingerprint", "MorganFingerprint"]
for fingerprint in fingerprints: # using the existing fingerprints in turn
    kcfmat.calc_fingerprints(fingerprint=fingerprint)
    X = kcfmat.fps_mat()
    y = classes
    print(fingerprint, ", length= ", len(X[0]))

    !date
    result = []
    for trial in range(10): # iterates 10 times
        clfs = Classifiers()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4) # generating training and test datasets
        for name, clf in clfs.classifiers: #  calling the defined classifiers in turn
            try:
                clf.fit(X_train, y_train) #  learning
                score1 = clf.score(X_train, y_train) # calculating the accuracy score for the training data
                score2 = clf.score(X_test, y_test) # calculating the accuracy score for the test data
                result.append([name, score1, score2]) # storing the result
            except:
                continue

    df_result = pd.DataFrame(result, columns=['classifier', 'train', 'test']) # Preparing the result table
    !date
    # grouping the results by classifiers and calculating the average of the accuracy score, followed by sorting
    df_result_mean = df_result.groupby('classifier').mean() #.sort_values('test', ascending=False)
    # calculating the standard deviations for the use of error bars
    errors = df_result.groupby('classifier').std()
    errors.columns=['train_err', 'test_err']

    # depicting the bar charts
    display(pd.concat([df_result_mean['train'], errors['train_err'], df_result_mean['test'], errors['test_err']], axis=1).sort_values('test', ascending=False))
    df_result_mean = df_result_mean.sort_values('test', ascending=False)
    errors.columns=['train', 'test']
    df_result_mean.plot(title=fingerprint, kind='bar', alpha=0.5, grid=True, yerr=errors, ylim=[0, 1])
```

    PatternFingerprint , length=  1024
    Mon Oct 15 07:34:44 UTC 2018
    Mon Oct 15 07:36:52 UTC 2018

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>train_err</th>
      <th>test</th>
      <th>test_err</th>
    </tr>
    <tr>
      <th>classifier</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Extra Tree</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.813043</td>
      <td>0.018062</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.807609</td>
      <td>0.015731</td>
    </tr>
    <tr>
      <th>Gaussian Process</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.805435</td>
      <td>0.015276</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.968909</td>
      <td>0.008461</td>
      <td>0.794293</td>
      <td>0.020418</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.988182</td>
      <td>0.005505</td>
      <td>0.791848</td>
      <td>0.022195</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.993636</td>
      <td>0.004791</td>
      <td>0.779891</td>
      <td>0.013915</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.844545</td>
      <td>0.014999</td>
      <td>0.766576</td>
      <td>0.016968</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.999455</td>
      <td>0.000878</td>
      <td>0.757880</td>
      <td>0.017350</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.891455</td>
      <td>0.007070</td>
      <td>0.755707</td>
      <td>0.010828</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.746467</td>
      <td>0.019217</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.796182</td>
      <td>0.084907</td>
      <td>0.701630</td>
      <td>0.059890</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.682182</td>
      <td>0.012591</td>
      <td>0.676630</td>
      <td>0.031404</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.647455</td>
      <td>0.012841</td>
      <td>0.644293</td>
      <td>0.023774</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.644364</td>
      <td>0.014601</td>
      <td>0.642935</td>
      <td>0.021822</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.999455</td>
      <td>0.000878</td>
      <td>0.642120</td>
      <td>0.039317</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.564402</td>
      <td>0.111159</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.611273</td>
      <td>0.025350</td>
      <td>0.527717</td>
      <td>0.029954</td>
    </tr>
  </tbody>
</table>
</div>

    LayeredFingerprint , length=  2048
    Mon Oct 15 07:37:13 UTC 2018
    Mon Oct 15 07:40:35 UTC 2018

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>train_err</th>
      <th>test</th>
      <th>test_err</th>
    </tr>
    <tr>
      <th>classifier</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Extra Tree</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.830707</td>
      <td>0.017473</td>
    </tr>
    <tr>
      <th>Gaussian Process</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.830707</td>
      <td>0.014831</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.997818</td>
      <td>0.002235</td>
      <td>0.829891</td>
      <td>0.011684</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>0.998909</td>
      <td>0.001271</td>
      <td>0.828804</td>
      <td>0.014265</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.978545</td>
      <td>0.003716</td>
      <td>0.820380</td>
      <td>0.026047</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.987455</td>
      <td>0.005518</td>
      <td>0.808967</td>
      <td>0.020934</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.944545</td>
      <td>0.020894</td>
      <td>0.800543</td>
      <td>0.020544</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.999091</td>
      <td>0.001286</td>
      <td>0.799457</td>
      <td>0.019545</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.852545</td>
      <td>0.010121</td>
      <td>0.793478</td>
      <td>0.016304</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.928000</td>
      <td>0.007865</td>
      <td>0.783967</td>
      <td>0.025012</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.993818</td>
      <td>0.005085</td>
      <td>0.753804</td>
      <td>0.026998</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.747554</td>
      <td>0.020353</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.718909</td>
      <td>0.023113</td>
      <td>0.692391</td>
      <td>0.021158</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.678000</td>
      <td>0.009484</td>
      <td>0.671467</td>
      <td>0.017208</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.762727</td>
      <td>0.042608</td>
      <td>0.666848</td>
      <td>0.039279</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.645091</td>
      <td>0.011777</td>
      <td>0.641848</td>
      <td>0.017601</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.566304</td>
      <td>0.065671</td>
    </tr>
  </tbody>
</table>
</div>

    RDKFingerprint , length=  2048
    Mon Oct 15 07:40:59 UTC 2018
    Mon Oct 15 07:44:36 UTC 2018

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>train_err</th>
      <th>test</th>
      <th>test_err</th>
    </tr>
    <tr>
      <th>classifier</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gaussian Process</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.825543</td>
      <td>0.019960</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.813315</td>
      <td>0.018343</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.812500</td>
      <td>0.016604</td>
    </tr>
    <tr>
      <th>Extra Tree</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.801087</td>
      <td>0.022254</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.987273</td>
      <td>0.003534</td>
      <td>0.800272</td>
      <td>0.013692</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.794837</td>
      <td>0.013328</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.987091</td>
      <td>0.006088</td>
      <td>0.785326</td>
      <td>0.019086</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.850909</td>
      <td>0.010808</td>
      <td>0.774185</td>
      <td>0.019234</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.944545</td>
      <td>0.056447</td>
      <td>0.772011</td>
      <td>0.048954</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.934727</td>
      <td>0.010408</td>
      <td>0.759511</td>
      <td>0.012169</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.992364</td>
      <td>0.005800</td>
      <td>0.757065</td>
      <td>0.025076</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.816000</td>
      <td>0.010420</td>
      <td>0.747011</td>
      <td>0.017909</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.803818</td>
      <td>0.035146</td>
      <td>0.728533</td>
      <td>0.045749</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.697283</td>
      <td>0.026227</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.719091</td>
      <td>0.021585</td>
      <td>0.681250</td>
      <td>0.028674</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.647273</td>
      <td>0.014520</td>
      <td>0.638587</td>
      <td>0.021701</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>0.999273</td>
      <td>0.000939</td>
      <td>0.577989</td>
      <td>0.064395</td>
    </tr>
  </tbody>
</table>
</div>

    MorganFingerprint , length=  2048
    Mon Oct 15 07:44:53 UTC 2018
    Mon Oct 15 07:47:58 UTC 2018

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>train_err</th>
      <th>test</th>
      <th>test_err</th>
    </tr>
    <tr>
      <th>classifier</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gaussian Process</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.855435</td>
      <td>0.013176</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.999091</td>
      <td>0.001286</td>
      <td>0.844837</td>
      <td>0.022019</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.842120</td>
      <td>0.023462</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833967</td>
      <td>0.019739</td>
    </tr>
    <tr>
      <th>Extra Tree</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.826630</td>
      <td>0.017508</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.977636</td>
      <td>0.021599</td>
      <td>0.818207</td>
      <td>0.027218</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.987636</td>
      <td>0.004354</td>
      <td>0.813315</td>
      <td>0.016948</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.853273</td>
      <td>0.017898</td>
      <td>0.810870</td>
      <td>0.014673</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.954364</td>
      <td>0.008241</td>
      <td>0.803804</td>
      <td>0.021196</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.915818</td>
      <td>0.010569</td>
      <td>0.777989</td>
      <td>0.025717</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.775272</td>
      <td>0.026066</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.972364</td>
      <td>0.004093</td>
      <td>0.749457</td>
      <td>0.023334</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.987273</td>
      <td>0.007761</td>
      <td>0.654076</td>
      <td>0.044762</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.645091</td>
      <td>0.016900</td>
      <td>0.641848</td>
      <td>0.025259</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.645091</td>
      <td>0.016900</td>
      <td>0.641848</td>
      <td>0.025259</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.645091</td>
      <td>0.016900</td>
      <td>0.641848</td>
      <td>0.025259</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.504076</td>
      <td>0.112736</td>
    </tr>
  </tbody>
</table>
</div>

![png](https://github.com/KCF-Convoy/kcfconvoy/raw/develop/docs/machine_learning_files/machine_learning_17_8.png)

![png](https://github.com/KCF-Convoy/kcfconvoy/raw/develop/docs/machine_learning_files/machine_learning_17_9.png)

![png](https://github.com/KCF-Convoy/kcfconvoy/raw/develop/docs/machine_learning_files/machine_learning_17_10.png)

![png](https://github.com/KCF-Convoy/kcfconvoy/raw/develop/docs/machine_learning_files/machine_learning_17_11.png)

## Calculating the KCF matrix

- Before using `.calc_kcf_matrix()`, the users only have a set of independent KCF vectors.
- The method `.calc_kcf_matrix()` enables the users to obtain a KCF matrix.

```python
!date
kcfmat.calc_kcf_matrix()
!date
```

    Mon Oct 15 07:48:00 UTC 2018
    Mon Oct 15 07:53:11 UTC 2018

- The naive KCF matrix `.all_mat` contains all possible chemical substructures, which yields a huge sparse matrix.

```python
kcfmat.all_mat.shape
```

    (918, 315558)

- The users can use the KCF matrix `.mat` in which rare chemical substructures (the observed frequency are lesser than the threshold) are removed.

```python
kcfmat.mat.shape
```

    (918, 37152)

## Selecting important features for the objective classification

- The original KCF matrix is still too huge. KCF Convoy provides the `.feature_selection()` method to select the important features for the classification of the user's objective. After the feature selection, the selected matrix can be accessed by `.selected_mat()`

```python
!date
y = classes
kcfmat.feature_selection(y, classifier=RandomForestClassifier())
X = kcfmat.selected_mat()
!date
```

    Mon Oct 15 08:02:00 UTC 2018
    Mon Oct 15 08:02:25 UTC 2018

```python
X.shape
```

    (918, 2048)

## Supervised binary classification using KCF-S

- Notice that the predicive performance is better than the best results by the other existing chemical fingerprints.

```python
!date
print("KCF-S", ", length= ", len(X[0]))
result = []
for trial in range(10): # iterates 10 times
    clfs = Classifiers()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4) # generating training and test datasets
    for name, clf in clfs.classifiers: #  calling the defined classifiers in turn
        try:
            clf.fit(X_train, y_train) #  learning
            score1 = clf.score(X_train, y_train) # calculating the accuracy score for the training data
            score2 = clf.score(X_test, y_test) # calculating the accuracy score for the test data
            result.append([name, score1, score2]) # storing the result
        except:
            continue

df_result = pd.DataFrame(result, columns=['classifier', 'train', 'test']) # Preparing the result table
!date
# grouping the results by classifiers and calculating the average of the accuracy score, followed by sorting
df_result_mean = df_result.groupby('classifier').mean() #.sort_values('test', ascending=False)
# calculating the standard deviations for the use of error bars
errors = df_result.groupby('classifier').std()
errors.columns=['train_err', 'test_err']

# depicting the bar charts
display(pd.concat([df_result_mean['train'], errors['train_err'], df_result_mean['test'], errors['test_err']], axis=1).sort_values('test', ascending=False))
df_result_mean = df_result_mean.sort_values('test', ascending=False)
errors.columns=['train', 'test']
df_result_mean.plot(title="KCF-S", kind='bar', alpha=0.5, grid=True, yerr=errors, ylim=[0, 1])
```

    Mon Oct 15 08:02:25 UTC 2018
    KCF-S , length=  2048
    Mon Oct 15 08:05:06 UTC 2018

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>train_err</th>
      <th>test</th>
      <th>test_err</th>
    </tr>
    <tr>
      <th>classifier</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>0.996727</td>
      <td>0.001434</td>
      <td>0.851630</td>
      <td>0.017480</td>
    </tr>
    <tr>
      <th>Extra Tree</th>
      <td>0.997273</td>
      <td>0.001286</td>
      <td>0.843750</td>
      <td>0.017622</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.971636</td>
      <td>0.002600</td>
      <td>0.839674</td>
      <td>0.016604</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.995818</td>
      <td>0.001497</td>
      <td>0.834511</td>
      <td>0.017160</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.986182</td>
      <td>0.004552</td>
      <td>0.825815</td>
      <td>0.014909</td>
    </tr>
    <tr>
      <th>Gaussian Process</th>
      <td>0.997273</td>
      <td>0.001286</td>
      <td>0.825000</td>
      <td>0.012166</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.948364</td>
      <td>0.010149</td>
      <td>0.811685</td>
      <td>0.015691</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.996909</td>
      <td>0.001227</td>
      <td>0.802446</td>
      <td>0.013680</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.997091</td>
      <td>0.000939</td>
      <td>0.791576</td>
      <td>0.027625</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.912545</td>
      <td>0.022358</td>
      <td>0.791033</td>
      <td>0.021452</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.857273</td>
      <td>0.009968</td>
      <td>0.787772</td>
      <td>0.010442</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.963818</td>
      <td>0.007295</td>
      <td>0.783424</td>
      <td>0.019345</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.808909</td>
      <td>0.009975</td>
      <td>0.777174</td>
      <td>0.016405</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.997273</td>
      <td>0.001286</td>
      <td>0.773370</td>
      <td>0.030856</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>0.997273</td>
      <td>0.001286</td>
      <td>0.755707</td>
      <td>0.023357</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.716545</td>
      <td>0.011124</td>
      <td>0.693750</td>
      <td>0.018477</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.652364</td>
      <td>0.009342</td>
      <td>0.655163</td>
      <td>0.017064</td>
    </tr>
  </tbody>
</table>
</div>

    <matplotlib.axes._subplots.AxesSubplot at 0x7f66088baa90>

![png](https://github.com/KCF-Convoy/kcfconvoy/raw/develop/docs/machine_learning_files/machine_learning_28_3.png)

## Depicting the chemical substructures for interpretation

- One of the advantages of KCF-S is its interpretability, since the users can identify which feature corresponds to which substructure.
- First, the selected features that are important for the classification can be obtained as the following way:

```python
print(kcfmat.selected_features)
```

    [24474, 5174, 13671, 8706, 667, 29962, 9499, 19649, 23222, 14767, 23765, 8264, 6686, 7285, 28291, 27143, 33191, 30630, 7902, 5726, 18041, 10075, 4394, 18113, 1661, 1797, 16044, 13004, 9908, 28327, 1079, 23576, 9432, 33743, 7347, 10467, 17972, 23279, 12048, 4670, 30898, 26987, 23971, 17050, 12201, 20741, 36230, 13293, 17595, 2906, 2350, 10677, 32429, 14469, 23785, 19796, 14162, 34796, 33970, 28715, 12879, 1393, 14374, 20040, 281, 23554, 10335, 16515, 2993, 2492, 6140, 15092, 20571, 28646, 33940, 19005, 1770, 10820, 18531, 17709, 24065, 5919, 15034, 18673, 3880, 20280, 3251, 9185, 25411, 2356, 21645, 15809, 1055, 25450, 36880, 12774, 20240, 12450, 31004, 8554, 31911, 13869, 11277, ...]

- By using one of the indices shown above, the users can see the name of the substructure, the compounds containing the substructure as shown below. The substructure are highted in the picture.

```python
print(kcfmat.strs[35802])
kcfmat.draw_cpds(kcfstringidx=35802)
```

    C8x-C8x-C8y-C8x-C8x-C8y-C1y-C1x-C8y-C8y-C1y-C8y-C8y-C8y-C8x-C8x

![png](https://github.com/KCF-Convoy/kcfconvoy/raw/develop/docs/machine_learning_files/machine_learning_32_1.png)

- The occurrances of the substructures in the respective compounds can be shown as below:

```python
kcfmat.mat[:, 35802]
```

    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...)

- The users can also compare the distributions of the specified substructures in the binary classes. In the case shown below, the specified substructures have different distributions, which provides a help to classify herbicides from others.

```python
dist1 = []
dist2 = []
for x, y in zip(kcfmat.mat[:, 35802], classes):
    if y == 1:
        dist1.append(x)
    else:
        dist2.append(x)
plt.hist([dist1, dist2], alpha=0.5, normed=True, label=["class 1", "class 0"])
plt.legend()
plt.show()
```

![png](https://github.com/KCF-Convoy/kcfconvoy/raw/develop/docs/machine_learning_files/machine_learning_36_0.png)
