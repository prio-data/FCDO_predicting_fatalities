
from viewser import Queryset, Column
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.decomposition import PCA


def SummarizeTable(dfname,df):
    print(f"{dfname}: A dataset with {len(df.columns)} columns, with "
      f"data between t = {min(df.index.get_level_values(0))} "
      f"and {max(df.index.get_level_values(0))}; "
      f"{len(np.unique(df.index.get_level_values(1)))} units."
     )

def FetchTable(Queryset, name):
    df = Queryset.fetch().astype(float)
    df.name = name
    SummarizeTable(name,df)
    Data = {
            'Name': name,
            'df': df
        }
    return(Data)

def FetchData(run_id):
    print('Fetching data using querysets; returns as list of dictionaries containing datasets')
    Datasets = []
    if run_id == 'Fatalities001':
        Datasets.append(FetchTable((Queryset("hh_fatalities_ged_ln_ultrashort", "country_month")),'baseline'))
        Datasets.append(FetchTable((Queryset("hh_fatalities_ged_acled_ln", "country_month")),'conflictlong_ln'))
        Datasets.append(FetchTable((Queryset("fat_cm_conflict_history", "country_month")),'conflict_ln'))
        Datasets.append(FetchTable((Queryset("fat_cm_conflict_history_exp", "country_month")),'conflict_nolog'))
        Datasets.append(FetchTable((Queryset("hh_fatalities_wdi_short", "country_month")),'wdi_short'))
        Datasets.append(FetchTable((Queryset("hh_fatalities_vdem_short", "country_month")),'vdem_short'))
        Datasets.append(FetchTable((Queryset("hh_topic_model_short", "country_month")),'topics_short'))
        Datasets.append(FetchTable((Queryset("hh_broad", "country_month")),'broad'))
        Datasets.append(FetchTable((Queryset("hh_prs", "country_month")),'prs'))
        Datasets.append(FetchTable((Queryset("hh_greatest_hits", "country_month")),'gh'))
        Datasets.append(FetchTable((Queryset("hh_20_features", "country_month")),'hh20'))
        Datasets.append(FetchTable((Queryset("hh_all_features", "country_month")),'all_features'))

        # PCA
        Standard_features = ['ln_ged_sb_dep','ln_ged_sb', 'decay_ged_sb_5', 'decay_ged_os_5', 'splag_1_decay_ged_sb_5', 'wdi_sp_pop_totl']

        sources = []
        af = {
            'name': 'all features',
            'dataset': Datasets[11]['df'],
            'n_comp': 20
        }
        sources.append(af)
        topics = {
            'name': 'topics',
            'dataset': Datasets[6]['df'],
            'n_comp': 10
        }
        sources.append(topics)
        vdem = {
            'name': 'vdem',
            'dataset': Datasets[5]['df'],
            'n_comp': 15
        }
        sources.append(vdem)
        wdi = {
            'name': 'wdi',
            'dataset': Datasets[4]['df'],
            'n_comp': 15
        }
        sources.append(wdi)

        EndOfPCAData = 516
        for source in sources:
            source = PCA(source, Standard_features,EndOfPCAData)

        Data = {
            'Name': 'pca_all',
            'df': af['result']
        }
        Datasets.append(Data)

        Data = {
            'Name': 'pca_topics',
            'df': topics['result']
        }
        Datasets.append(Data)

        Data = {
            'Name': 'pca_vdem',
            'df': vdem['result']
        }
        Datasets.append(Data)

        Data = {
            'Name': 'pca_wdi',
            'df': wdi['result']
        }
        Datasets.append(Data)
        

        return(Datasets)

def PCA(source, Standard_features, EndOfPCAData):
    df = source['dataset'].loc[121:EndOfPCAData].copy()
    df = df.replace([np.inf, -np.inf], 0) 
    df = df.fillna(0)
    pca = decomposition.PCA(n_components=source['n_comp'])
    pca.fit(df)
    df1 = pd.DataFrame(pca.transform(df))

    print(source['name'],pca.explained_variance_ratio_)

    print(pca.singular_values_)
    df2 = source['dataset'][Standard_features].loc[121:EndOfPCAData].copy()
    source['result'] = pd.concat([df2, df1.set_index(df2.index)], axis=1)
    colnames = Standard_features.copy()
    for i in range(source['n_comp']):
        colname = 'pc' + str(i+1)
        colnames.append(colname)
    source['result'].columns = colnames
    source['result'].head()
    return(source)


def find_index(dicts, key, value):
    class Null: pass
    for i, d in enumerate(dicts):
        if d.get(key, Null) == value:
            return i
    else:
        raise ValueError('no dict with the key and value combination found')

def RetrieveFromList(Datasets,name):
    return Datasets[find_index(Datasets, 'Name', name)]['df']
 
