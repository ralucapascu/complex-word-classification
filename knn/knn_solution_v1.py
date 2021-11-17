import pandas as pd
import numpy as np
from dale_chall import DALE_CHALL
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

dtypes = {"sentence": "string", "token": "string", "complexity": "float64"}

train = pd.read_excel('train.xlsx', dtype=dtypes, keep_default_na=False) #filepath, putem da ca parametru si sheet_name(List[str]), dtype -> data type for data or columns
test = pd.read_excel('test.xlsx', dtype=dtypes, keep_default_na=False) #keep_default_na -> whether or not to include the default NaN values when parsing data

def is_title(word):
    return int(word.istitle())

def is_dale_chall(word):
    return int(word in DALE_CHALL);

def get_word_structure_features(word):
    features = []
    features.append(is_title(word))
    features.append(is_dale_chall(word))
    return np.array(features)

def get_corpus_feature(corpus):
    d = {"bible": [0], "europarl": [1], "biomed": [2]}
    return d[corpus]

def featurize(row):
    word = row["token"]
    corpus = row["corpus"]
    features = []
    features.extend(get_word_structure_features(word))
    features.extend(get_corpus_feature(corpus))
    return features
    
def featurize_df(df):
    nr_of_features = len(featurize(df.iloc[0]))
    nr_of_rows = len(df)
    print("Numarul de features:", nr_of_features)
    print("Numarul de exemple:", nr_of_rows)
    features = np.zeros((nr_of_rows, nr_of_features))
    i = 0
    for index, row in df.iterrows():
        row_ftrs = featurize(row)
        features[index, :] = row_ftrs
        i += 1
    return features

def get_submission():
    X = featurize_df(train)
    Y = train["complex"].values
    X_test = featurize_df(test)

    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X, Y)
    preds = model.predict(X_test)

    preds = model.predict(X_test)
    print(preds)

    test_id = np.arange(7663,9001) 

    np.savetxt("submisie_Kaggle_7nn_v1_fresh.csv",np.stack((test_id,preds)).T,fmt="%d",delimiter=',',header="id,complex",comments="")
    
def get_balanced_acc_score():
    kf = KFold(n_splits = 10)
    kf.get_n_splits(train)
    Y = train["complex"].values
    balanced_acc_score = 0
    k_values = [7]

    for k in k_values:
        for train_index, test_index in kf.split(train):
            X_train, X_test = featurize_df(train.iloc[train_index]), featurize_df(train.iloc[test_index])
            Y_train, Y_test = Y[train_index], Y[test_index]
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, Y_train)
            preds = model.predict(X_test)
            balanced_acc_score += balanced_accuracy_score(Y_test, preds)
        print(k, balanced_acc_score/10)
    
# get_balanced_acc_score()

get_submission()


# print(train)
# k_values = range(1, 31)
# print(type(train.iloc[k_values]))
# print(train.iloc[k_values])

# featurize_df(train.iloc[k_values])
