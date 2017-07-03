import re
import pandas as pd
import numpy as np


def get_title(name):
    ''' convert name to title group
    '''
    def title_map(title):
        ''' map less frequent titles to groups
        '''
        nobs = ['Countess.', 'Don.', 'Dona.', 'Jonkheer.', 'Lady.', 'Sir.', 'Dr.']
        miss = ['Mlle.', 'Mme.', 'Ms.']
        mltr = ['Capt.', 'Col.', 'Major.']
        if title in nobs:
            return 'prestige'
        if title in miss:
            return 'miss'
        if title in mltr:
            return 'military'
        return title.lower().strip('.')
    try:
        return title_map(re.findall('\w+[.]', name)[0])
    except:
        return np.NaN


def family_size(df):
    ''' return number of family members accompanying
        passenger
    '''
    return df.sibsp + df.parch + 1


def impute_by_title(df, column):
    ''' calculate the median for numeric valued column
        after grouping by title
        return impute function to replace np.nan with
        the corresponding title's median
    '''
    title_median = df.groupby('title')[column].median(skipna=True)
    def median_impute(row):
        if pd.isnull(row[column]):
            return title_median[row.title]
        return row[column]
    return median_impute


def fare_category(df):
    ''' cut fare category into three slices
        test set contains NA values, hence the impute step
    '''
    fare_wo_na = df.apply(impute_by_title(df, 'fare'), axis=1)
    return pd.qcut(fare_wo_na, 4, labels=range(4))


def prep_features(filepath):
    ''' prep all features
    '''
    df = (pd.read_csv(filepath)
            .rename(columns=str.lower)
            .assign(title=lambda df: df.name.apply(get_title))
            .assign(
                fam_size=lambda df: family_size(df),
                fare_cat=lambda df: fare_category(df),
                age_imputed= lambda df:(
                    df.apply(impute_by_title(df, 'age'), axis=1)
                ))
            .drop(['name', 'cabin', 'ticket', 'fare', 'sibsp', 'parch', 'age'], axis=1))
    return (df.drop(['sex', 'title', 'embarked'], axis=1)
              .join(pd.get_dummies(df[['sex', 'title', 'embarked']])))