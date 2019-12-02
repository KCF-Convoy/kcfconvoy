# Example usage of KCF-Convoy for machine learning

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
import kcfconvoy as kcf # KCF-Convoy library
from kcfconvoy import Classifiers # a set of classifiers for machine learning
```

## Input compound structures in BRITE

- Input compound structures using KEGG BRITE https://www.genome.jp/kegg/brite.html

Example file can be obtained by clicking the <b>[Download htext]</b> link in https://www.genome.jp/kegg-bin/get_htext?ko01002.keg . KCF-Convoy provides a method `.input_from_brite()` to input compounds from KEGG BRITE. RDKit may generate many warnings, which are because of the chemical structures defined in KEGG.

```python
# Please download "br08007.keg" file by clicking "Download htext" link from https://www.genome.jp/kegg-bin/get_htext?br08007.keg
# and put it in the appropriate directory.
!date
brite = './kegg/br08007.keg'
kcfmat = kcf.KCFmat()
kcfmat.input_from_brite(brite)
!date
```

    Mon Oct 15 11:06:28 UTC 2018


    RDKit WARNING: [11:06:29]  S group MUL ignored on line 75
    RDKit WARNING: [11:06:30] WARNING: Omitted undefined stereo
    RDKit WARNING: [11:07:38] WARNING: Metal was disconnected; Proton(s) added/removed

    Mon Oct 15 11:17:02 UTC 2018

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
      'C18498',
      'C18723',
      'C18568',
      'C10929',

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

- The users can classify compounds into two groups by specifiying the class shown above. KCF-Convoy provides a method `.brite_class()` to give a binary classification describing whether or not a group belongs to the specified group. In the case provided below, `(3, 'B:Herbicides', 365)` is specified, and the user will deal with the classification of herbicides from others.

```python
classes = kcfmat.brite_class(3)
```

- Check the classification results, which will be used as a training data

```python
print(classes)
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

## Supervised binary classification using existing chemical fingerprints

- KCF-Convoy provides `Classifiers` for a number of classifiers, and `.calc_fingerprints()` for a set of existing chemical fingerprints (defined in RDKit).

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
    Mon Oct 15 11:17:09 UTC 2018
    Mon Oct 15 11:18:53 UTC 2018

<div>
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
      <td>0.811413</td>
      <td>0.011543</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.810054</td>
      <td>0.018136</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.964909</td>
      <td>0.003741</td>
      <td>0.805707</td>
      <td>0.017293</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.988364</td>
      <td>0.004632</td>
      <td>0.798913</td>
      <td>0.021588</td>
    </tr>
    <tr>
      <th>Gaussian Process</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.794565</td>
      <td>0.011614</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.992000</td>
      <td>0.003657</td>
      <td>0.786141</td>
      <td>0.017844</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.846545</td>
      <td>0.013530</td>
      <td>0.767935</td>
      <td>0.019729</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.999091</td>
      <td>0.000958</td>
      <td>0.766576</td>
      <td>0.023392</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.891636</td>
      <td>0.010185</td>
      <td>0.754891</td>
      <td>0.017461</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.742935</td>
      <td>0.027120</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.806000</td>
      <td>0.109930</td>
      <td>0.714130</td>
      <td>0.072053</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.680182</td>
      <td>0.010301</td>
      <td>0.679891</td>
      <td>0.024366</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.644182</td>
      <td>0.011847</td>
      <td>0.651902</td>
      <td>0.026422</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.640182</td>
      <td>0.014881</td>
      <td>0.649185</td>
      <td>0.022241</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.997818</td>
      <td>0.002064</td>
      <td>0.644837</td>
      <td>0.023795</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.582609</td>
      <td>0.139929</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.606909</td>
      <td>0.023904</td>
      <td>0.526359</td>
      <td>0.021891</td>
    </tr>
  </tbody>
</table>
</div>

    LayeredFingerprint , length=  2048
    Mon Oct 15 11:19:14 UTC 2018
    Mon Oct 15 11:22:41 UTC 2018

<div>
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
      <td>0.999636</td>
      <td>0.000767</td>
      <td>0.826630</td>
      <td>0.022908</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>0.999636</td>
      <td>0.000767</td>
      <td>0.814130</td>
      <td>0.016859</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.998545</td>
      <td>0.001434</td>
      <td>0.813315</td>
      <td>0.023552</td>
    </tr>
    <tr>
      <th>Extra Tree</th>
      <td>0.999636</td>
      <td>0.000767</td>
      <td>0.811141</td>
      <td>0.013389</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.980000</td>
      <td>0.007273</td>
      <td>0.807609</td>
      <td>0.013051</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.988364</td>
      <td>0.006193</td>
      <td>0.795924</td>
      <td>0.018227</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.856727</td>
      <td>0.010767</td>
      <td>0.789674</td>
      <td>0.016010</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.937818</td>
      <td>0.032815</td>
      <td>0.787228</td>
      <td>0.021399</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.999273</td>
      <td>0.001271</td>
      <td>0.785598</td>
      <td>0.016427</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.927091</td>
      <td>0.011091</td>
      <td>0.782609</td>
      <td>0.019215</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.992545</td>
      <td>0.005031</td>
      <td>0.744022</td>
      <td>0.024500</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.999636</td>
      <td>0.000767</td>
      <td>0.743750</td>
      <td>0.035269</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.724545</td>
      <td>0.021585</td>
      <td>0.690761</td>
      <td>0.034056</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.768909</td>
      <td>0.047152</td>
      <td>0.672554</td>
      <td>0.038627</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.678182</td>
      <td>0.015989</td>
      <td>0.667935</td>
      <td>0.024163</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.647091</td>
      <td>0.015295</td>
      <td>0.638859</td>
      <td>0.022859</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>0.999636</td>
      <td>0.000767</td>
      <td>0.581522</td>
      <td>0.065129</td>
    </tr>
  </tbody>
</table>
</div>

    RDKFingerprint , length=  2048
    Mon Oct 15 11:23:03 UTC 2018
    Mon Oct 15 11:26:49 UTC 2018

<div>
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
      <td>0.999818</td>
      <td>0.000575</td>
      <td>0.838859</td>
      <td>0.013254</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>0.999818</td>
      <td>0.000575</td>
      <td>0.833424</td>
      <td>0.019002</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.999818</td>
      <td>0.000575</td>
      <td>0.831522</td>
      <td>0.018739</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.985636</td>
      <td>0.002491</td>
      <td>0.827174</td>
      <td>0.015699</td>
    </tr>
    <tr>
      <th>Extra Tree</th>
      <td>0.999818</td>
      <td>0.000575</td>
      <td>0.813859</td>
      <td>0.011186</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.999818</td>
      <td>0.000575</td>
      <td>0.804891</td>
      <td>0.014873</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.968727</td>
      <td>0.019522</td>
      <td>0.801630</td>
      <td>0.020576</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.990182</td>
      <td>0.004128</td>
      <td>0.791576</td>
      <td>0.015214</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.837091</td>
      <td>0.008791</td>
      <td>0.782337</td>
      <td>0.028427</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.819273</td>
      <td>0.014776</td>
      <td>0.761685</td>
      <td>0.017566</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.928909</td>
      <td>0.009169</td>
      <td>0.759783</td>
      <td>0.017943</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.988545</td>
      <td>0.006184</td>
      <td>0.759239</td>
      <td>0.029160</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.813273</td>
      <td>0.023815</td>
      <td>0.745652</td>
      <td>0.035643</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.999818</td>
      <td>0.000575</td>
      <td>0.729348</td>
      <td>0.020504</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.719273</td>
      <td>0.027443</td>
      <td>0.694293</td>
      <td>0.030076</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.642545</td>
      <td>0.013061</td>
      <td>0.645652</td>
      <td>0.019520</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>0.999818</td>
      <td>0.000575</td>
      <td>0.637772</td>
      <td>0.058926</td>
    </tr>
  </tbody>
</table>
</div>

    MorganFingerprint , length=  2048
    Mon Oct 15 11:27:04 UTC 2018
    Mon Oct 15 11:30:12 UTC 2018

<div>
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
      <td>0.844293</td>
      <td>0.012554</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.998182</td>
      <td>0.001212</td>
      <td>0.835870</td>
      <td>0.012823</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.834239</td>
      <td>0.015157</td>
    </tr>
    <tr>
      <th>Multi-Layer Perceptron</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.833424</td>
      <td>0.014496</td>
    </tr>
    <tr>
      <th>Extra Tree</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.815489</td>
      <td>0.014963</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.988364</td>
      <td>0.003114</td>
      <td>0.806250</td>
      <td>0.015951</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.954545</td>
      <td>0.010739</td>
      <td>0.800543</td>
      <td>0.019436</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.977091</td>
      <td>0.022222</td>
      <td>0.800000</td>
      <td>0.021366</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.861818</td>
      <td>0.012121</td>
      <td>0.796739</td>
      <td>0.019919</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.772011</td>
      <td>0.018627</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.916182</td>
      <td>0.010194</td>
      <td>0.770924</td>
      <td>0.012223</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.973455</td>
      <td>0.006595</td>
      <td>0.756250</td>
      <td>0.024441</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.987818</td>
      <td>0.011878</td>
      <td>0.697011</td>
      <td>0.046156</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.646364</td>
      <td>0.011443</td>
      <td>0.639946</td>
      <td>0.017103</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.646364</td>
      <td>0.011443</td>
      <td>0.639946</td>
      <td>0.017103</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.646364</td>
      <td>0.011443</td>
      <td>0.639946</td>
      <td>0.017103</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.557337</td>
      <td>0.100543</td>
    </tr>
  </tbody>
</table>
</div>

![png](machine_learning_files/machine_learning_17_8.png)

![png](machine_learning_files/machine_learning_17_9.png)

![png](machine_learning_files/machine_learning_17_10.png)

![png](machine_learning_files/machine_learning_17_11.png)

## Calculating the KCF matrix

- Before using `.calc_kcf_matrix()`, the users only have a set of independent KCF vectors.
- The method `.calc_kcf_matrix()` enables the users to obtain a KCF matrix.

```python
!date
kcfmat.calc_kcf_matrix()
!date
```

    Mon Oct 15 11:30:14 UTC 2018
    Mon Oct 15 11:35:22 UTC 2018

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

- The original KCF matrix is still too huge. KCF-Convoy provides the `.feature_selection()` method to select the important features for the classification of the user's objective. After the feature selection, the selected matrix can be accessed by `.selected_mat()`

```python
!date
y = classes
kcfmat.feature_selection(y, classifier=RandomForestClassifier())
X = kcfmat.selected_mat()
!date
```

    Mon Oct 15 11:35:23 UTC 2018
    Mon Oct 15 11:40:45 UTC 2018

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

    Mon Oct 15 11:40:46 UTC 2018
    KCF-S , length=  2048
    Mon Oct 15 11:43:17 UTC 2018

<div>
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
      <td>0.997091</td>
      <td>0.001533</td>
      <td>0.859239</td>
      <td>0.017367</td>
    </tr>
    <tr>
      <th>Extra Tree</th>
      <td>0.997455</td>
      <td>0.001271</td>
      <td>0.849457</td>
      <td>0.023488</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.995636</td>
      <td>0.001533</td>
      <td>0.840217</td>
      <td>0.019461</td>
    </tr>
    <tr>
      <th>Gradient Boosting</th>
      <td>0.967636</td>
      <td>0.004983</td>
      <td>0.834783</td>
      <td>0.013725</td>
    </tr>
    <tr>
      <th>Gaussian Process</th>
      <td>0.997455</td>
      <td>0.001271</td>
      <td>0.831793</td>
      <td>0.016626</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.985455</td>
      <td>0.005353</td>
      <td>0.827174</td>
      <td>0.017433</td>
    </tr>
    <tr>
      <th>Linear SVM</th>
      <td>0.997091</td>
      <td>0.001533</td>
      <td>0.816848</td>
      <td>0.018966</td>
    </tr>
    <tr>
      <th>AdaBoost</th>
      <td>0.946182</td>
      <td>0.010746</td>
      <td>0.805978</td>
      <td>0.016613</td>
    </tr>
    <tr>
      <th>Nearest Neighbors</th>
      <td>0.860000</td>
      <td>0.010141</td>
      <td>0.794293</td>
      <td>0.017093</td>
    </tr>
    <tr>
      <th>Stochastic Gradient Descent</th>
      <td>0.887818</td>
      <td>0.038855</td>
      <td>0.792663</td>
      <td>0.017566</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.951636</td>
      <td>0.007531</td>
      <td>0.789946</td>
      <td>0.014608</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.997455</td>
      <td>0.001271</td>
      <td>0.785054</td>
      <td>0.012966</td>
    </tr>
    <tr>
      <th>Linear Discriminant Analysis</th>
      <td>0.997455</td>
      <td>0.001271</td>
      <td>0.779620</td>
      <td>0.024953</td>
    </tr>
    <tr>
      <th>RBF SVM</th>
      <td>0.815455</td>
      <td>0.017922</td>
      <td>0.770924</td>
      <td>0.017045</td>
    </tr>
    <tr>
      <th>Quadratic Discriminant Analysis</th>
      <td>0.996545</td>
      <td>0.003476</td>
      <td>0.748913</td>
      <td>0.050663</td>
    </tr>
    <tr>
      <th>Sigmoid SVM</th>
      <td>0.692909</td>
      <td>0.014931</td>
      <td>0.666848</td>
      <td>0.020262</td>
    </tr>
    <tr>
      <th>Polynomial SVM</th>
      <td>0.660545</td>
      <td>0.009623</td>
      <td>0.637228</td>
      <td>0.012567</td>
    </tr>
  </tbody>
</table>
</div>

    <matplotlib.axes._subplots.AxesSubplot at 0x7ff27584db70>

![png](machine_learning_files/machine_learning_28_3.png)

## Depicting the chemical substructures for interpretation

- One of the advantages of KCF-S is its interpretability, since the users can identify which feature corresponds to which substructure.
- First, the selected features that are important for the classification can be obtained as the following way:

```python
print(kcfmat.selected_features)
```

    [10998, 16219, 9125, 4253, 14986, 5298, 25449, 28596, 5080, 21818, 10759, 14491, 16228, 10828, 6557, 15339, 3917, 17956, 11252, 26580, 14914, 30085, 14563, 9670, 641, 16940, 433, 18054, 29856, 4041, 3377, 31018, 19936, 32700, 20715, 8656, 9673, 33748, 1881, 28460, 17238, 24634, 7048, 19082, 30401, 24191, 7082, 28236, 7526, 14596, 1716, 24149, 15311, 7411, 9215, 9105, 21754, 29900, 28694, 23119, 26263, 9113, 32751, 13870, 5365, 16413, 20566, 5935, 3809, 4360, 13569, 33372, 18821, 14774, 22856, 12256, 22213, 17124, 16873, 11797, 22947, 6519, 26913, 6993, 9664, 8704, 29811, 27630, 18526, 6461, 2612, 33997, 7633, 35635, 4566, 29937, 12304, 19017, 2794, 24592, 13929, 14554, 34720, 17160, 22003, 16652, 24560, 19599, 7792, 23382, 4070, 25528, 23370, 13517, 32061, 17998, 29845, 154, 33408, 25799, 24579, 24589, 35823, 17016, 30967, 27600, 4513, 1487, 29114, 6496, 2144, 14685, 13480, 18145, 37011, 24623, 6467, 35016, 16803, 31152, 8081, 29872, 6473, 22786, 29280, 16471, 13461, 28913, 14097, 18860, 3073, 16650, 24789, 23635, 12121, 7662, 3272, 9926, 7837, 17023, 35227, 33421, 22795, 15255, 29453, 34763, 11753, 21348, 8459, 34978, 30818, 5702, 4301, 21308, 7906, 7084, 7290, 16051, 13266, 20421, 14829, 35095, 3461, 28479, 19395, 33529, 31888, 9523, 14124, 27460, 27568, 34644, 22209, 18468, 8132, 13876, 36965, 12463, 33691, 26456, 3873, 26405, 7917, 34928, 6038, 23377, 23424, 34904, 10917, 15104, 25573, 9336, 14923, 15632, 10988, 30704, 10952, 4178, 34174, 28481, 28980, 30237, 10427, 4337, 2918, 17852, 22529, 8646, 25504, 19394, 11988, 4964, 20199, 23032, 32776, 1914, 14290, 23197, 16434, 1586, 35214, 8858, 3115, 26603, 37041, 14411, 15041, 33173, 36255, 7739, 5187, 32659, 2087, 6444, 15987, 241, 6544, 12685, 947, 14985, 24237, 10356, 35690, 8753, 17152, 33878, 31932, 23127, 21144, 36373, 15554, 2531, 8693, 8330, 25608, 3437, 3727, 28806, 31071, 18153, 34705, 25945, 29766, 31756, 5052, 17165, 4859, 8317, 35793, 31119, 1426, 12629, 14934, 13166, 5904, 31721, 8485, 14881, 14971, 7078, 35256, 34156, 26076, 37001, 9949, 7065, 12524, 12893, 23586, 6609, 22765, 36043, 31493, 21292, 23692, 1782, 9339, 33319, 13629, 26995, 22592, 17954, 22153, 460, 13006, 33703, 20821, 10181, 4356, 21788, 16248, 25672, 12469, 6820, 25226, 39, 20992, 31581, 14162, 1148, 23805, 8936, 23047, 22387, 5600, 20244, 23225, 378, 26439, 27802, 31146, 32250, 25624, 11683, 14659, 13655, 31221, 9720, 18926, 17180, 28790, 1759, 19634, 12166, 3244, 28208, 16250, 3579, 3944, 35050, 15983, 8591, 33540, 29691, 30479, 29634, 31235, 29217, 36229, 27234, 7300, 29714, 14482, 31361, 3182, 10267, 23270, 15559, 33443, 18522, 23435, 22339, 1724, 24102, 11243, 29430, 28399, 9856, 34055, 13452, 24050, 15687, 25456, 23461, 10682, 34753, 27508, 25120, 7248, 22246, 34046, 22893, 23647, 26979, 30307, 9007, 28374, 7351, 5421, 1302, 18835, 35270, 9773, 28749, 30178, 28220, 34990, 9714, 28014, 10410, 15488, 30783, 5357, 30494, 23072, 12517, 2615, 18164, 5025, 30305, 489, 1931, 7933, 19410, 9212, 29367, 29890, 15345, 15742, 7001, 11734, 26294, 29546, 31269, 36727, 31515, 6005, 15518, 36230, 28721, 12092, 6681, 6067, 6847, 23671, 33636, 28099, 23637, 36638, 15763, 26792, 9152, 34390, 24059, 22089, 20521, 27052, 24703, 10753, 30583, 4591, 15516, 14654, 36385, 4950, 14697, 22059, 36926, 31166, 26368, 13641, 4912, 22854, 10112, 27793, 9416, 32938, 15984, 17794, 30000, 14594, 28650, 28178, 14592, 35761, 12855, 29305, 35135, 33267, 25902, 5228, 12069, 30584, 16364, 21080, 20843, 21477, 26723, 32219, 23024, 914, 13448, 18715, 3619, 36362, 25154, 10576, 18634, 18913, 11990, 21124, 415, 36096, 34184, 28660, 34641, 11617, 14235, 32905, 15679, 30480, 15859, 21994, 17687, 36134, 23380, 3856, 23286, 34311, 5273, 25574, 32609, 21140, 17702, 24085, 24168, 4722, 31043, 11636, 23125, 35636, 33627, 3000, 4982, 14637, 24126, 6910, 20093, 4355, 24200, 2922, 18166, 16328, 26398, 25973, 5619, 17850, 692, 23094, 25313, 6201, 2286, 9499, 18719, 10787, 1312, 18377, 23680, 16633, 8361, 3447, 22177, 34145, 22312, 21013, 18435, 35279, 28608, 15218, 13412, 12840, 28280, 12883, 29519, 29983, 3554, 19422, 1650, 8752, 24576, 31522, 4321, 32384, 11575, 19200, 34871, 30653, 29493, 29590, 16169, 21079, 19286, 2763, 31325, 20237, 32474, 19850, 7156, 15682, 18582, 31874, 31639, 15657, 36735, 4224, 22040, 7856, 24395, 20996, 2234, 31615, 32913, 7536, 22971, 32655, 14795, 20157, 23016, 8027, 12346, 12676, 8155, 8247, 17993, 11824, 19061, 6202, 30686, 17311, 36568, 21985, 2413, 35709, 36423, 30014, 958, 29646, 18167, 34759, 29944, 15099, 17422, 35340, 2284, 30941, 31079, 12740, 7167, 18971, 18540, 36019, 34599, 22586, 1944, 30336, 4781, 11308, 25453, 28767, 6146, 1372, 29041, 17067, 36621, 21709, 30445, 7576, 7373, 869, 14972, 5533, 27722, 7135, 23609, 23584, 112, 32686, 14942, 21978, 1037, 29966, 18725, 12057, 3764, 25191, 20697, 9805, 34907, 12747, 2527, 262, 2209, 360, 31052, 25106, 2288, 22504, 11042, 650, 3228, 29601, 12189, 26660, 295, 33799, 11699, 24335, 15908, 34402, 2039, 3683, 6334, 34286, 7718, 5860, 29223, 36932, 194, 26568, 19398, 24529, 29680, 33277, 2881, 21286, 7843, 33948, 17352, 2636, 30280, 36855, 8963, 35171, 14213, 32187, 15033, 7826, 15299, 30035, 9151, 1974, 33082, 28489, 8198, 10633, 8389, 1340, 3445, 34130, 4090, 9332, 8486, 14295, 7247, 34369, 33333, 26700, 9719, 21367, 33058, 32580, 33630, 711, 24294, 6932, 1101, 4899, 24127, 4405, 34885, 17770, 14817, 21830, 37122, 4862, 12513, 6402, 30544, 16677, 10300, 3374, 16460, 23524, 10382, 18284, 7039, 22034, 16077, 18309, 9232, 26464, 4065, 21464, 14664, 160, 8218, 16835, 17193, 29311, 8374, 34943, 35629, 7999, 1880, 651, 12148, 35847, 32527, 28364, 31061, 36859, 15880, 26972, 21752, 27562, 26043, 5276, 7801, 2996, 27088, 26245, 12140, 12473, 1286, 15958, 11577, 36102, 28417, 18920, 16741, 19495, 13198, 35940, 5808, 26196, 13710, 18094, 3186, 15684, 7867, 4016, 9266, 1130, 6319, 34205, 13557, 36576, 21715, 28901, 25274, 27001, 22426, 18186, 14809, 2124, 19307, 21117, 22833, 35297, 2986, 30470, 9745, 27181, 4840, 7333, 7127, 7214, 26538, 27130, 20814, 11212, 35826, 28634, 33846, 11188, 30132, 23895, 5449, 1110, 3730, 3040, 24279, 818, 23650, 23536, 21488, 19766, 2408, 25749, 35444, 36956, 7227, 9281, 22754, 7102, 31971, 15018, 23388, 13933, 21428, 36578, 18307, 15469, 171, 5107, 658, 15827, 19940, 20947, 126, 22518, 29481, 5914, 33251, 30970, 31802, 36450, 7527, 162, 24393, 18193, 28205, 13065, 36002, 2142, 28824, 22288, 18697, 18198, 24236, 1067, 17921, 6226, 6481, 2590, 7424, 30537, 29078, 15166, 30547, 11247, 27029, 25435, 7841, 15219, 11493, 1470, 20845, 33889, 29893, 16917, 15228, 36990, 14038, 15418, 369, 22725, 11177, 15485, 32065, 7062, 7317, 20267, 16699, 30853, 3381, 10196, 2566, 24701, 26069, 23139, 6684, 7385, 36202, 23576, 2400, 27012, 17896, 21248, 3172, 8514, 27023, 2338, 35476, 24313, 29886, 25179, 23757, 15253, 34212, 17765, 30084, 6802, 37060, 20749, 34829, 21793, 1113, 30270, 9, 4142, 3888, 22175, 6049, 29795, 12173, 11468, 31613, 17171, 26189, 5823, 8246, 14634, 32534, 27849, 36939, 6730, 5418, 25001, 34843, 15775, 13177, 13585, 12397, 9341, 1800, 11934, 20550, 32857, 13956, 6586, 21812, 16606, 35551, 3035, 4646, 35036, 34827, 24517, 16635, 24340, 6427, 13104, 32870, 29809, 12815, 21201, 34065, 33113, 36646, 8011, 28342, 10023, 19487, 10765, 17510, 8490, 31546, 9882, 22787, 31557, 218, 21490, 26295, 21503, 22409, 34531, 7974, 32513, 21410, 30083, 2735, 10832, 15394, 12970, 36399, 24694, 30413, 23953, 35894, 14667, 3834, 35842, 14192, 10653, 35323, 4527, 15570, 14562, 633, 35739, 31298, 3042, 17916, 35994, 271, 34407, 16337, 7354, 28027, 28740, 22635, 12272, 34844, 18136, 2194, 12844, 16459, 30934, 33847, 9505, 5500, 8476, 9906, 31998, 22468, 18679, 24953, 34498, 6046, 11095, 21970, 8478, 11983, 13362, 30759, 4102, 31668, 3771, 20922, 25903, 9150, 9834, 24094, 25617, 30911, 19680, 25691, 11191, 33967, 13513, 33578, 25879, 24961, 32755, 12494, 20736, 11079, 8250, 33876, 11015, 8443, 30592, 16666, 6697, 4146, 27178, 31050, 17312, 28011, 15178, 34382, 19935, 23152, 17242, 21244, 17211, 30179, 10219, 30026, 30794, 35469, 30507, 31202, 9496, 2179, 7978, 28098, 24617, 19065, 31612, 12557, 1502, 34039, 3652, 22429, 3847, 30188, 21571, 26242, 26366, 16868, 20487, 23673, 32164, 12094, 10519, 10509, 22960, 23036, 4525, 24800, 30578, 19406, 15514, 863, 1871, 3664, 7137, 16266, 37093, 13672, 25333, 15039, 12062, 10685, 2113, 21560, 1577, 7944, 15856, 17883, 19424, 23846, 11869, 4942, 23070, 12716, 4230, 20700, 29594, 20030, 10035, 15059, 3726, 11422, 5082, 16578, 28158, 32900, 1895, 9671, 15150, 3003, 20917, 35070, 18918, 32519, 28961, 10668, 35533, 30512, 20603, 9148, 12178, 10954, 19182, 21391, 18532, 28512, 2755, 11560, 14362, 7745, 19700, 4952, 29756, 30013, 31318, 36853, 12728, 17258, 18320, 28187, 23791, 20409, 14215, 31599, 14453, 11842, 7550, 3266, 20127, 6827, 35873, 20849, 29834, 3538, 18586, 8020, 2645, 15171, 29009, 23311, 18436, 32382, 17940, 27450, 5102, 11066, 18274, 13777, 18142, 25068, 21454, 29825, 7186, 26899, 7506, 1514, 14502, 14500, 20155, 14265, 29777, 22922, 24045, 1817, 10525, 9739, 30594, 917, 4064, 13030, 17149, 24823, 32588, 9586, 11269, 15275, 23483, 29278, 11344, 2197, 32233, 48, 19052, 34463, 3733, 15429, 2772, 30138, 19735, 17836, 14922, 28542, 14009, 22563, 5068, 30415, 9259, 9010, 12084, 36987, 4044, 28333, 31891, 4105, 4127, 31912, 33841, 8474, 4575, 14347, 30256, 325, 11769, 20202, 6378, 21484, 7683, 25512, 27848, 31951, 29406, 6781, 20188, 4037, 32267, 20645, 5157, 16582, 21062, 2188, 24940, 25541, 35388, 17146, 8164, 17119, 7426, 6561, 16403, 30373, 11201, 2943, 34621, 36497, 22771, 30914, 12530, 26089, 2415, 14792, 3966, 13576, 15246, 33164, 29486, 16787, 15142, 1793, 36690, 1331, 14267, 6030, 27163, 9063, 27277, 19230, 5484, 18986, 6805, 24967, 9822, 4772, 6626, 2048, 26414, 34748, 409, 11199, 29451, 13548, 29500, 2054, 23741, 14935, 13152, 15813, 16600, 2475, 19971, 28018, 33190, 5967, 6635, 3949, 25653, 29131, 8154, 22459, 11353, 11601, 3901, 8205, 8053, 20376, 15546, 31256, 24101, 29607, 13079, 20820, 2331, 22669, 4284, 31077, 19513, 35621, 14125, 8350, 22001, 17362, 26733, 11246, 19083, 18459, 7602, 8116, 10003, 36511, 5608, 17692, 463, 34081, 19201, 7016, 20625, 13471, 15323, 29374, 20186, 28876, 11949, 31603, 8305, 32651, 27420, 29831, 23561, 5210, 25491, 30666, 26195, 21303, 30943, 5580, 8251, 20332, 28960, 2221, 11007, 13209, 5735, 11632, 36217, 11693, 12529, 14898, 35259, 26597, 19374, 33825, 26191, 24327, 13637, 11694, 14005, 24913, 13948, 17465, 16062, 28034, 19993, 12996, 30890, 23738, 22638, 34633, 25936, 10734, 36535, 22901, 8487, 18741, 35997, 33067, 29027, 27886, 30698, 36746, 3963, 7081, 2933, 20965, 588, 8266, 19288, 10320, 8113, 17279, 13617, 31506, 549, 20130, 17697, 9594, 34293, 29076, 24602, 32122, 14825, 2424, 26352, 7993, 8700, 10565, 1291, 20618, 12156, 2644, 23771, 16528, 13673, 8724, 16098, 9365, 13940, 17657, 7788, 18453, 14756, 33999, 36842, 19397, 9898, 28290, 23966, 12224, 21443, 12655, 27203, 5472, 9683, 6153, 26889, 28336, 1494, 27465, 9635, 22405, 15691, 5310, 7284, 32861, 9539, 35915, 2023, 6846, 35671, 4541, 35494, 12582, 24534, 12465, 31870, 34396, 31182, 19304, 16550, 15780, 3176, 25, 32847, 21049, 11849, 16175, 9698, 698, 22874, 15633, 26688, 27880, 13824, 35523, 20722, 6434, 2720, 23287, 28136, 4814, 25564, 18343, 7542, 21228, 32893, 23689, 25112, 24253, 24, 23643, 19585, 13193, 29302, 34897, 17776, 34727, 35667, 30349, 23837, 19034, 27893, 4756, 17641, 4842, 24675, 17941, 21822, 24381, 18929, 7336, 26400, 2970, 2602, 2080, 2783, 26230, 10631, 31275, 28248, 35147, 663, 27474, 3457, 6335, 13939, 27419, 10850, 30098, 2711, 7198, 1200, 16715, 20701, 10141, 23790, 32919, 10790, 20388, 6325, 12894, 11713, 370, 27179, 18946, 13897, 2049, 24133, 9356, 616, 35892, 9137, 3195, 28616, 17538, 25529, 23964, 29170, 30108, 15615, 7529, 28121, 29020, 10652, 12509, 13430, 36502, 15114, 7047, 4472, 34341, 30447, 34856, 17203, 11985, 12616, 14146, 19881, 31463, 30058, 23035, 9762, 17553, 16740, 19176, 1210, 20461, 9317, 23815, 35442, 20915, 16993, 25366, 1644, 19392, 8089, 16295, 8557, 2065, 26590, 30519, 7282, 36360, 2517, 13980, 25359, 16213, 29683, 14653, 29623, 30654, 26473, 29255, 337, 11739, 12212, 19430, 26281, 10538, 14927, 35669, 10241, 3049, 12942, 10657, 3796, 33651, 8699, 31151, 149, 289, 30900, 9358, 4597, 13819, 3887, 9944, 5925, 13823, 32367, 6200, 21858, 7395, 32463, 28832, 21634, 17872, 23978, 23868, 15978, 10407, 34674, 28563, 4744, 924, 458, 22143, 18878, 19224, 34255, 7954, 12342, 36109, 26362, 32573, 36601, 16801, 12279, 26442, 1162, 36441, 27014, 19553, 5799, 26561, 11991, 9107, 32569, 35687, 9835, 15186, 12847, 19895, 8907, 30718, 34159, 34833, 22262, 11917, 31869, 27712, 1485, 15610, 16040, 27046, 17712, 5445, 11696, 6817, 11175, 219, 19536, 22825, 16222, 25923, 34549, 9974, 24002, 21909, 3597, 33583, 6280, 35763, 22902, 15960, 5329, 20816, 12334, 15935, 31577, 17495, 1318, 30199, 28373, 6836, 8086, 24428, 11388, 6251, 1268, 22969, 3941, 9420, 15336, 33086, 19419, 33015, 32844, 18619, 24659, 28252, 19898, 18325, 3371, 7562, 13477, 11966, 32345, 25069, 34445, 17837, 28175, 36295, 20923, 30422, 17459, 342, 31353, 30489, 2334, 4673, 13671, 15262, 22079, 36632, 5796, 17741, 13025, 29443, 6570, 26959, 10590, 33271, 18887, 36274, 17055, 21825, 8197, 16206, 11768, 3092, 8799, 18445, 1618, 19567, 31375, 8558, 17799, 25197, 31974, 5443, 12446, 2256, 13203, 2699, 12432, 21226, 32578, 14056, 9792, 31284, 20480, 33611, 28594, 8382, 10076, 4945, 332, 19102, 27440, 31807, 30005, 15731, 34917, 26432, 9830, 20949, 23963, 13003, 11231, 16406, 14638, 1004, 34113, 15300, 4603, 2705, 27104, 12306, 18263, 24671, 15725, 29616, 13162, 20189, 7158, 24627, 19759, 28509, 12878, 27338, 26655, 15553, 18648, 22249, 1684, 14811]

- By using one of the indices shown above, the users can see the name of the substructure, the compounds containing the substructure as shown below. The substructure are highted in the picture.

```python
print(kcfmat.strs[10998])
kcfmat.draw_cpds(kcfstringidx=10998)
```

    C-C-N-C-O

![png](machine_learning_files/machine_learning_32_1.png)

- The occurrances of the substructures in the respective compounds can be shown as below:

```python
kcfmat.mat[:, 10998]
```

    array([ 2,  0,  0,  0,  4,  2,  2,  2,  0,  1,  0,  2,  2,  2,  3,  1,  0,
            2,  4,  2,  1,  2,  4,  4,  4,  2,  4,  2,  3,  2,  0,  2,  3,  6,
            0,  0,  0,  0,  0,  3,  6,  0,  4,  6,  2,  2,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 18,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  8,  2,  6,  2,  9,  6,
            8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  1,  0,  2,
            0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  5,  0,  0,  0,  3,  0,  0,  0,  0,  0,  3,  2,  4,  0,  5,  0,
            0,  2,  4,  2,  2,  2,  3,  5,  5,  5,  2,  2,  3,  2,  2,  4,  0,
            2,  4,  0,  2,  3,  4,  2,  3,  0,  0,  4,  4,  0,  0,  0,  2,  2,
            0,  0,  0,  0,  0,  0,  5,  4,  8,  8,  4,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
            1,  0,  0,  0,  0,  0,  0,  0,  4,  0,  4,  0,  0,  0,  0,  0,  0,
            3,  2,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  2,  2,  2,  2,  2,
            5,  6,  2,  2,  2,  2,  5,  2,  2,  4,  3,  0,  0,  2,  0,  0,  0,
            3,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
            0,  1,  1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  3,  2,  2,  3,
            2,  2,  2,  2,  2,  4,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  4,  1,  0,  7,  7,  9,  4,  1,  0,  3,  0,  0,  0,  0,
            3,  0,  4,  2,  0,  3,  2,  0,  0,  1,  2,  0,  0,  0,  0,  0,  0,
            4,  0,  0,  0,  3,  1,  1,  0,  0,  2,  0,  4,  7,  8,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,
            0,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  2,
            2,  0,  0,  0,  0,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,
            0,  3,  0,  6,  0,  0,  0,  0,  0,  1,  0,  0,  4,  0,  1,  5,  0,
            0,  6,  2,  0,  0,  0,  3,  2,  0,  0,  0,  3,  0,  0,  1,  4,  4,
            0,  3,  3,  3,  3,  3,  3,  3,  3,  0,  1,  0,  3,  3,  3,  3,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,
            0,  0,  2,  0,  4,  2,  0,  0,  2,  1,  4,  2,  2,  2,  2,  2,  0,
            3,  0,  0,  0,  0,  8,  8,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  2,  1,  0,  0,  0,
            0,  2,  4,  5,  2,  0,  0,  4,  2,  3,  2,  5,  3,  4,  2,  0,  0,
            0,  4,  4, 10,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  1,  2,  4,  2,  2,  2,  2,  2,  2,  0,  2,  2,  2,  2,  5,
            4,  2,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  2,
            2,  0,  2,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  7,  0,  6,
            2,  0,  8,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  1,
            2,  0,  0,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  6,  0,  0,  0,  0,  1,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])

- The users can also compare the distributions of the specified substructures in the binary classes. In the case shown below, the specified substructures have different distributions, which provides a help to classify herbicides from others.

```python
dist1 = []
dist2 = []
for x, y in zip(kcfmat.mat[:, 10998], classes):
    if y == 1:
        dist1.append(x)
    else:
        dist2.append(x)
plt.hist([dist1, dist2], alpha=0.5, normed=True, label=["class 1", "class 0"])
plt.legend()
plt.show()
```

![png](machine_learning_files/machine_learning_36_0.png)

```python

```
