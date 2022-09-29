'''
Read in data (json) and display statstic on JJIF
'''

import datetime as dt
import json
import os
import re
import requests

from requests.auth import HTTPBasicAuth
import pandas as pd
from pandas import json_normalize
import plotly.express as px
import plotly.graph_objs as pg
import streamlit as st
import pycountry_convert as pc
import numpy as np

# for the name matching
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct  # Leading Juice for us

# to build the singelton
from datastore import DataStore

# the supported age_divisions
AGE_INP = ["U16", "U18", "U21", "Adults", "U14", "U12", "U10", "U15"]
# preselected age_divisions
AGE_SEL = ["U16", "U18", "U21", "Adults"]

# the supported disciplines
DIS_INP = ["Duo", "Show", "Jiu-Jitsu", "Fighting", "Contact"]
# preselected disciplines
DIS_SEL = ["Duo", "Show", "Jiu-Jitsu", "Fighting"]

# continents
CONT_INP = ["Europe", "Pan America", "Africa", "Asia", "Oceania"]

# types of events
EVENT_TYPE = ['National Championship', 'Continental Championship',
              'World Championship', 'A Class Tournament',
              'World Games / Combat Games']
EVENT_INP = ['Continental Championship',
             'World Championship', 'A Class Tournament',
             'World Games / Combat Games']

COLOR_MAP = {"Jiu-Jitsu": 'rgb(243, 28, 43)',
             "Fighting": 'rgb(0,144,206)',
             "Duo": 'rgb(211,211,211)',
             "Show": 'rgb(105,105,105)',
             "Contact": 'rgb(255,255,255)'}

COLOR_MAP_CON = {"Europe": 'rgb(243, 28, 43)',
                 "Asia": 'rgb(0,144,206)',
                 "Pan America": 'rgb(211,211,211)',
                 "Africa": 'rgb(105,105,105)',
                 "Oceania": 'rgb(255,255,255)'}


def read_in_iso():
    ''' Read in file
     - HELPER FUNCTION TO READ IN A CSV FILE and convert NOC code to ISO

    '''
    inp_file = pd.read_csv("Country,NOC,ISOcode.csv", sep=',')
    ioc_iso = inp_file[
        ['NOC', 'ISO code']
    ].set_index('NOC').to_dict()['ISO code']

    return ioc_iso


def read_in_catkey():
    ''' Read in file
     - HELPER FUNCTION
     Reads in a csv  and convert catergory ids to catergoy names

    '''
    inp_file = pd.read_csv("catID_name.csv", sep=';')
    key_map_inp = inp_file[
        ['cat_id', 'name']
    ].set_index('cat_id').to_dict()['name']

    return key_map_inp


def data_setting():
    ''' The sidebar elements for the selection
    '''

    st.sidebar.image("https://i0.wp.com/jjeu.eu/wp-content/uploads/2018/08/jjif-logo-170.png?fit=222%2C160&ssl=1",
                     use_column_width='always')
    mode_in = st.sidebar.selectbox('Please select your mode',
                                   ('History', 'Single Event', 'Countries'))
    dstart_in = st.sidebar.date_input("From", dt.date(2010, 1, 1))
    dend_in = st.sidebar.date_input("To", dt.date.today())

    age_select_in = st.sidebar.multiselect('Select the age divisions',
                                           AGE_INP,
                                           AGE_SEL)
    dis_select_in = st.sidebar.multiselect('Select the disciplines',
                                           DIS_INP,
                                           DIS_SEL)
    para_in = st.sidebar.selectbox('Para setting',
                                   ('Include', 'Exclude','Only'),
                                   help='Include = Include Para in statistic,\
                                   Exclude = Exclude Para in statistic , \
                                   Only = Shows only Para disciplines')
    cont_select_in = st.sidebar.multiselect('Select the continent',
                                            CONT_INP,
                                            CONT_INP)
    evtt_select_in = st.sidebar.multiselect('Select the event type',
                                            EVENT_TYPE,
                                            EVENT_INP)

    return age_select_in, dis_select_in, cont_select_in, dstart_in, \
        dend_in, evtt_select_in, mode_in, para_in


def get_events(dstart, dend, evtt_select, user, password):
    '''
    enter ther start and

    Parameters
    ----------
    dstart

    '''

    start_str = dstart.strftime("%Y%m%d")
    end_str = dend.strftime("%Y%m%d")
    uri = "https://www.sportdata.org/ju-jitsu/rest/events/ranked/" + start_str + "/" + end_str + "/"

    response = requests.get(uri, auth=HTTPBasicAuth(user, password))

    d = response.json()
    df2 = json_normalize(d)
    df2 = df2[['id', 'country_code', 'name', 'eventtype', 'startDate']]
    df2['startDate'] = df2['startDate'].str[:10]


    # one needs to get wevent which are not in the db
    # f2 = open("all_events.json", "r")
    # returns JSON object as
    # a dictionary
    df_wg =[ {'id': 'TWG2013', 'country_code': 'COL',
              'name': 'The World Games 2013',
              'eventtype': 'World Games / Combat Games',
              'startDate': '2013-07-30'},
             {'id': 'YWCh2016', 'country_code': 'ESP',
              'name': 'U21 World Championships 2016 Madrid',
              'eventtype': 'World Championship',
              'startDate': '2016-03-18'},
             {'id': 'POp2016', 'country_code': 'FRA',
              'name': 'Paris Open 2016',
              'eventtype': 'A Class Tournament',
              'startDate': '2016-04-30'},
             {'id': 'EOC2016', 'country_code': 'BEL',
              'name': 'European Open Championship 2016',
              'eventtype': 'A Class Tournament',
              'startDate': '2016-06-04'},
             {'id': 'USO2016', 'country_code': 'USA',
              'name': 'US Open 2016',
              'eventtype': 'A Class Tournament',
              'startDate': '2016-07-09'},
             {'id': 'AFCH2016', 'country_code': 'RSA',
              'name': 'African Championship 2016',
              'eventtype': 'Continental Championship',
              'startDate': '2016-08-20'},
             {'id': 'Pam2016', 'country_code': 'PAN',
              'name': 'Pan American Championship 2016',
              'eventtype': 'Continental Championship',
              'startDate': '2016-08-26'},
             {'id': 'Go2016', 'country_code': 'GER',
              'name': 'German Open 2016',
              'eventtype': 'A Class Tournament',
              'startDate': '2016-09-24'},
              {'id': 'BO2016', 'country_code': 'GER',
              'name': 'Balkan Open 2016',
              'eventtype': 'A Class Tournament',
              'startDate': '2016-09-16'},
              {'id':'SAO2016', 'country_code': 'COL',
              'name': 'South American Open 2016',
              'eventtype': 'A Class Tournament',
              'startDate': '2016-10-18'},
              {'id':'WCh2016', 'country_code': 'POL',
              'name': 'World Championship 2016',
              'eventtype': 'World Championship',
              'startDate': '2016-11-25'},
              {'id':'ACh2016', 'country_code': 'POL',
              'name': 'Asian Championship 2016',
              'eventtype': 'Continental Championship',
              'startDate': '2016-12-09'},
              {'id': 'YWCH2017', 'country_code': 'GRE',
              'name': 'Youth World Championships 2017 Athens',
              'eventtype': 'World Championship',
              'startDate': '2017-03-17'},
              {'id': 'POp2017', 'country_code': 'FRA',
              'name': 'Paris Open 2017',
              'eventtype': 'A Class Tournament',
              'startDate': '2017-04-29'},
              {'id':'WCh2015', 'country_code': 'THA',
              'name': 'World Championship 2015',
              'eventtype': 'World Championship',
              'startDate': '2015-11-23'},
              {'id':'WCh2014', 'country_code': 'FRA',
              'name': 'World Championship 2014',
              'eventtype': 'World Championship',
              'startDate': '2014-11-28'},
              {'id':'WCh2012', 'country_code': 'AUT',
              'name': 'World Championship 2012',
              'eventtype': 'World Championship',
              'startDate': '2012-11-30'},
              {'id': 'Go2015', 'country_code': 'GER',
              'name': 'German Open 2015',
              'eventtype': 'A Class Tournament',
              'startDate': '2015-10-02'},
              {'id': 'ACh2015', 'country_code': 'RSA',
              'name': 'African Championship 2015',
              'eventtype': 'Continental Championship',
              'startDate': '2015-10-14'},
            ]

    df2 = df2.append(df_wg, ignore_index = True)            

    df2['startDate'] = pd.to_datetime(df2["startDate"]).dt.date
    df2 = df2[df2['startDate'].between(dstart, dend, inclusive=False)]
    df2 = df2[df2['eventtype'].isin(evtt_select)]

    
    return df2


def update_events(df_evts, age_select, dis_select, cont_select, evtt):

    frames = []
    if len(age_select) > 0 and len(dis_select) > 0 and len(cont_select) > 0 and len(evtt) > 0:
        with st.expander("Hide/include indivudual events"):
            evt_sel = df_evts['name'].unique().tolist()
            container = st.container()
            all = st.checkbox("Select all",value=True)
            if all:
                evtt_select = container.multiselect("Select the event:",
                                                    evt_sel,
                                                    evt_sel)
            else:
                evtt_select = container.multiselect("Select the event:",
                                                    evt_sel)
            df_evts = df_evts[df_evts['name'].isin(evtt_select)]

        # read in a all events and add to df
        events = df_evts['id'].tolist()

        my_bar = st.progress(0)
 
	# event loop
        with st.spinner('Recalculate events'):
            for i, val in enumerate(events):
                if ds.check_cache(val):
                    df_file = ds.get_data(val)
                    if len(df_file) > 0:
                        frames.append(df_file)
                else:
                    d = FileCheck(val, st.secrets['user'], st.secrets['password'],
                                  st.secrets['user1'],st.secrets['password1'], st.secrets['url'])
                    if len(d) > 0:
                        df_file = json_normalize(d['results'])
                        df_file = df_file[['category_id', 'category_name', 'country_code', 'rank', 'name']]
                        df_file['id'] = val
                        # add event dataframe to general data
                        ds.set_data(val, df_file) 
                        frames.append(df_file)
                    else:
                        ds.set_data(val, pd.DataFrame())
                my_bar.progress((i+1)/len(events))
    return frames


def FileCheck(numb, user, password, user1, password1, data_url):
    '''
    runs over all files in event list and get them from
    sportdata or from the local database

    '''
    # TODO:
    # build cache and check if dict is already in cache
    # dict is defined by numb (unique identifier)
    db_string = "curl -u " + user1+":"+password1+" "+data_url+numb+".json > "+numb+".json"
    os.system(db_string)
    d = {}
    if os.stat(numb+".json").st_size > 256:
        with open(numb+".json", "r") as f:
            d = json.load(f)   
            return d
    else:
        print("Error: File "+ numb +" does not appear to in local database. Try online")
        uri2 = 'https://www.sportdata.org/ju-jitsu/rest/event/resultscomplete/' + str(numb) + '/'
        response2 = requests.get(uri2, auth=HTTPBasicAuth(user, password))
        f3 = response2.json()
        
        if len(f3) <= 0:
            print("event " + numb +" is not online, skip event")
            d = {}        
            return d
        else:
            return f3

def conv_to_type(df, type_name, type_list):
    '''
    checks strings in data frames and
    replaces them with default input
    '''
    for x in type_list:
        df[type_name].where(~(df[type_name].str.contains(x)),
                            other=x, inplace=True)
    return df[type_name]


def ngrams(string, n=3):
    ''' used to check for simliar names'''
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    ''' force A and B as a CSR matrix.
     If they have already been CSR, there is no overhead'''
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)
    return csr_matrix((data, indices, indptr),shape=(M,N))


def get_matches_df(sparse_matrix, name_vector, top=100):
    '''unpacks the resulting sparse matrix'''
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if sparsecols.size > top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similarity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                        'right_side': right_side,
                        'similarity': similarity})

# Main programm starts here

#create data store
ds = DataStore()

IOC_ISO = read_in_iso()
key_map = read_in_catkey()
age_select, dis_select, cont_select, dstart, dend, evtt, mode, para_inp = data_setting()

df_evts = get_events(dstart, dend, evtt, st.secrets['user'], st.secrets['password'])
frames = update_events(df_evts, age_select, dis_select, cont_select, evtt)

if len(frames) == 0:
    st.write("please select at least one item in each category")
else:
    # cleanup of df
    df_ini = pd.concat(frames)
    df_ini['name'] = df_ini['name'].apply(lambda x: x.upper())
    df_ini['name'].replace("  ", " ", regex=True, inplace=True)
    df_ini['name'].replace("  ", " ", regex=True, inplace=True)
    # remove different characters in names
    df_ini['name'].replace("-", "/", regex=True, inplace=True)
    df_ini['name'].replace(" / ", "/", regex=True, inplace=True)
    df_ini['name'].replace(" /", "/", regex=True, inplace=True)
    df_ini['name'].replace("/ ", "/", regex=True, inplace=True)
    df_ini['name'].replace("Ö", "OE", regex=True, inplace=True)
    df_ini['name'].replace("Ä", "AE", regex=True, inplace=True)
    df_ini['name'].replace("Ü", "UE", regex=True, inplace=True)
    df_ini['name'].replace("Ć", "C", regex=True, inplace=True)
    df_ini['name'].replace("Š", "S", regex=True, inplace=True)
    df_ini['name'].replace("Ó", "O", regex=True, inplace=True)
    df_ini['name'].replace("Á", "A", regex=True, inplace=True)
    df_ini['name'].replace("Ñ", "A", regex=True, inplace=True)
    df_ini['name'].replace("Ï", "A", regex=True, inplace=True)
    df_ini['name'].replace("Í", "I", regex=True, inplace=True)
    df_ini['name'].replace("É", "E", regex=True, inplace=True)
    df_ini['name'].replace("Ő", "O", regex=True, inplace=True)
    df_ini['name'].replace("Č", "C", regex=True, inplace=True)
    df_ini['name'].replace("Ń", "N", regex=True, inplace=True)
    df_ini['name'].replace(",", " ", regex=True, inplace=True)
    df_ini['name'].replace("  ", " ", regex=True, inplace=True)
    df_ini['rank'] = df_ini['rank'].astype(int)
    df_ini['category_id'] = df_ini['category_id'].astype(int)

    # remove all categories which are not in key map and convert to hr name
    with st.expander("Show unsupported categories", expanded=False):
        st.write(df_ini[~df_ini['category_id'].isin(key_map.keys())])
    df_ini = df_ini[df_ini['category_id'].isin(key_map.keys())]
    df_ini['category_name'] = df_ini['category_id'].replace(key_map)

    # merge similar names
    cat_list = df_ini['category_name'].unique().tolist()

    # loop over all catergories
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

    with st.expander('Details on name matching', expanded=False):
        st.write('Similar names were matched to aviod double courning. This is based on:')
        st.write('https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e')
        similarity = st.number_input('similarity', min_value=0.4, 
                                     max_value=0.9, value=0.6,
                                     help="small number means more matches, high number only exact matches"
                                     )
    # create empty temporary list for events to fix names
    list_df_new = []
    
    for i in range(len(cat_list)):
        df_new = df_ini[df_ini['category_name'].str.contains(str(cat_list[i])) == True]

        # re-index the names column to continuous index starting at
        names_types = pd.Series(df_new['name'].values)

        if len(names_types) > 1:
            tf_idf_matrix = vectorizer.fit_transform(names_types)
            if len(names_types) > 4: 
                matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.4)
            else:
                matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 4, 0.4) 
            # store the  matches into new dataframe called matched_df
            matches_df = get_matches_df(matches, names_types, top=200)
            # For removing all exact matches
            matches_df = matches_df[matches_df['similarity'] < 0.99999]
            # create a mapping between names in form of a dict
            matches_df = matches_df[matches_df['similarity'] > similarity]
            dict_map = dict(zip(matches_df.left_side, matches_df.right_side))
            df_new.loc[:, 'name'] = df_new['name'].replace(dict_map)

            list_df_new.append(df_new)
            if len(dict_map) > 0:
                print('fixing ' + str(len(dict_map)) + ' issues with names')

    # overwrite existing df_ini with events with name issues fixed
    df_ini = pd.concat(list_df_new)

    # replace wrong country codes and make all ISO
    df_ini["country_code"].replace("RJF", "RUS", regex=True, inplace=True)
    df_ini["country_code"].replace("JIF", "LIE", regex=True, inplace=True)
    df_ini["country_code"].replace("ENG", "GBR", regex=True, inplace=True)
    df_ini['country_code'] = df_ini['country_code'].replace(IOC_ISO)
    df_ini['continent'] = df_ini['country_code'].apply(
                          lambda x: pc.country_alpha2_to_continent_code(x))
    df_ini['country'] = df_ini['country_code'].apply(
                        lambda x: pc.country_alpha2_to_country_name(x))
    df_ini['country_code'] = df_ini['country_code'].apply(
                             lambda x: pc.country_alpha2_to_country_name(x))
    df_ini['country_code'] = df_ini['country_code'].apply(
                             lambda x: pc.country_name_to_country_alpha3(x))
    df_ini['continent'] = df_ini['continent'].apply(
                          lambda x: pc.convert_continent_code_to_continent_name(x))
    df_ini['continent'].where(~(df_ini['continent'].str.contains("South America")),
                              other="Pan America", inplace=True)
    df_ini['continent'].where(~(df_ini['continent'].str.contains("North America")),
                              other="Pan America", inplace=True)
    # String comparison does not handle + well... replaced with p in csv and here replaced back
    df_ini['category_name'].replace("p", "+", regex=True, inplace=True)
    df_ini['cat_type'] = df_ini['category_name']
    df_ini['cat_type'] = conv_to_type(df_ini, 'cat_type', DIS_INP)

    df_ini['age_division'] = df_ini['category_name']
    df_ini['age_division'] = conv_to_type(df_ini, 'age_division', AGE_INP)

    # remove what is not selected
    df_ini = df_ini[df_ini['cat_type'].isin(dis_select)]
    df_ini = df_ini[df_ini['age_division'].isin(age_select)]
    df_ini = df_ini[df_ini['continent'].isin(cont_select)]
    if para_inp =='Exclude':
        df_ini = df_ini[~df_ini['category_name'].str.contains("Para")]
    elif para_inp =='Only':
        df_ini = df_ini[df_ini['category_name'].str.contains("Para")]
    else:
        print("Include Para")

    df_par = df_ini.copy()
    df_par = df_par.join(df_evts[['id','startDate']].set_index('id'), on='id') 
    df_min = df_par[['country', 'name', 'category_name', 'startDate']].groupby(['country', 'name', 'category_name']).min().reset_index()
    df_min.rename(columns={"startDate": "entryDate"}, inplace=True)

    df_max = df_par[['country', 'name', 'category_name', 'startDate']].groupby(['country', 'name', 'category_name']).max().reset_index()
    df_max.rename(columns={"startDate": "leavingDate"}, inplace=True)
    df_max['leavingDate'] = df_max['leavingDate'] + pd.offsets.DateOffset(years=2)
    df_total = pd.merge(df_min, df_max)

    st.write("in total ", len(df_total), "athletes")

    df_total['long_id'] = df_total['country'] + "_" + df_total['name'] + "_" + df_total['category_name']
    df_total['gender'] = df_total['category_name']
    df_total['gender'].where(~(df_total['gender'].str.contains("Men")),
                       other="Men", inplace=True)
    df_total['gender'].where(~(df_total['gender'].str.contains("Women")),
                       other="Women", inplace=True)
    df_total['gender'].where(~(df_total['gender'].str.contains("Mixed")),
                       other="Mixed", inplace=True)

    df_total['country_code'] = df_total['country'].apply(lambda x: pc.country_name_to_country_alpha2(x))

    df_total['continent'] = df_total['country_code'].apply(lambda x: pc.country_alpha2_to_continent_code(x))
    df_total['continent'] = df_total['continent'].apply(lambda x: pc.convert_continent_code_to_continent_name(x))
    df_total['country_code'] = df_total['country_code'].apply(lambda x: pc.country_alpha2_to_country_name(x))
    df_total['country_code'] = df_total['country_code'].apply(lambda x: pc.country_name_to_country_alpha3(x))
    df_total['continent'].where(~(df_total['continent'].str.contains("South America")),
                          other="Pan America", inplace=True)
    df_total['continent'].where(~(df_total['continent'].str.contains("North America")),
                          other="Pan America", inplace=True)

    df_total['cat_type'] = df_total['category_name']
    df_total['cat_type'] = conv_to_type(df_total, 'cat_type', DIS_INP)

    df_total['age_division'] = df_total['category_name']
    df_total['age_division'] = conv_to_type(df_total, 'age_division', AGE_INP)


    #merge the entires to a time range
    df_crosstab = (
       pd.crosstab(
        index=df_total['long_id'],
        columns=df_total['entryDate']
        ).sub(
            pd.crosstab(
                index=df_total['long_id'],
                columns=df_total['leavingDate']
                ),
            fill_value=0
            ).cumsum(axis=1)
        )
    df_crosstab.reset_index(inplace=True)

    df_time = pd.melt(
        df_crosstab,
        id_vars='long_id',
        value_vars=df_crosstab.columns[:-1],
        var_name='dates'
    )

    # discard all 0 values (no mutations)
    df_time = df_time[df_time['value'] > 0]
    # delete the 'value' column (now always 1)
    del df_time['value']

    df_time['temp_id'] = df_time['long_id'].str.split("_")
    df_time['country'] = df_time['temp_id'].apply(lambda x: x[0])
    df_time['name'] = df_time['temp_id'].apply(lambda x: x[1])
    df_time['category_name'] = df_time['temp_id'].apply(lambda x: x[2])
    df_time.drop('temp_id', inplace=True, axis=1)
    df_time['cat_type'] = df_time['category_name']
    df_time['cat_type'] = conv_to_type(df_time, 'cat_type', DIS_INP)

    df_time['age_division'] = df_time['category_name']
    df_time['age_division'] = conv_to_type(df_time, 'age_division', AGE_INP)

    df_time['country_code'] = df_time['country'].apply(lambda x: pc.country_name_to_country_alpha2(x))

    df_time['continent'] = df_time['country_code'].apply(lambda x: pc.country_alpha2_to_continent_code(x))
    df_time['continent'] = df_time['continent'].apply(lambda x: pc.convert_continent_code_to_continent_name(x))
    df_time['continent'].where(~(df_time['continent'].str.contains("South America")),
                          other="Pan America", inplace=True)
    df_time['continent'].where(~(df_time['continent'].str.contains("North America")),
                          other="Pan America", inplace=True)

    # start grapics here

    st.header('JJIF statistic')

    if mode =='History': 
        df_timeev = df_time[['dates', 'name', 'cat_type']].groupby(['dates', 'cat_type']).count().reset_index()
        fig1 = px.area(df_timeev, x='dates', y='name', color='cat_type', title="Time evolution of JJIF - Athletes (stacked))",
                      color_discrete_map=COLOR_MAP)
        fig1.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig1)

        fig1a = px.line(df_timeev, x='dates', y='name', color='cat_type', title="Time evolution of JJIF - Athletes",
                      color_discrete_map=COLOR_MAP)
        fig1a.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig1a)

        df_timeev_jjnos = df_time[['dates', 'country', 'continent']].groupby(['dates', 'continent']).nunique().reset_index()
        fig0 = px.area(df_timeev_jjnos, x='dates', y='country', color='continent', title="Time evolution of JJIF - JJNOs",
                      color_discrete_map=COLOR_MAP_CON)
        fig0.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig0)

        df_timeev_jjnos_dis = df_time[['dates', 'country', 'cat_type']].groupby(['dates', 'cat_type']).nunique().reset_index()
        fig0a = px.line(df_timeev_jjnos_dis, x='dates', y='country', color='cat_type', title="Time evolution of JJIF - JJNOs discipline",
                      color_discrete_map=COLOR_MAP)
        fig0a.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig0a)

        current_cat = st.checkbox('Show only currently active athletes', value=True)
        if current_cat: 
            df_total = df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]

        df_cats = df_total[['name','category_name','cat_type','continent']].groupby(['category_name','cat_type','continent']).count().reset_index()
        
        fig_cats = px.bar(df_cats, x="category_name", y="name", color="continent", title="Athletes per category", color_discrete_map=COLOR_MAP_CON)
        fig_cats.update_layout(xaxis={'categoryorder':'category ascending'})
        st.plotly_chart(fig_cats)

        df_cats_jjnos = df_total[['country','category_name','cat_type','continent']].groupby(['category_name','cat_type','continent']).nunique().reset_index()
        
        fig_cats_jjnos = px.bar(df_cats_jjnos, x="category_name", y="country", color="continent", title="JJNOs per category", color_discrete_map=COLOR_MAP_CON)
        fig_cats_jjnos.update_layout(xaxis={'categoryorder':'category ascending'})
        st.plotly_chart(fig_cats_jjnos)

        left_column, right_column = st.columns(2)
        with left_column:
            df_cat = pd.DataFrame()
            df_cat['cat_type'] = df_ini['cat_type'].value_counts().index
            df_cat['counts'] = df_ini['cat_type'].value_counts().values
            fig1 = px.pie(df_cat, values='counts',
                          names='cat_type', color='cat_type',
                          title='Discipline distribution',
                          color_discrete_map=COLOR_MAP)
            fig1.update_layout(legend=dict(
                               yanchor="top",
                               y=0.99,
                               xanchor="left",
                               x=0.01))
            st.plotly_chart(fig1,use_container_width=True)

        with right_column:
        	df_gender = pd.DataFrame()
        	df_gender['gender'] = df_total['gender'].value_counts().index
        	df_gender['counts'] = df_total['gender'].value_counts().values	
        	fig2 = px.pie(df_gender, values='counts', names='gender', color='gender',
                          color_discrete_map={"Women": 'rgb(243, 28, 43)',
                                              "Men": 'rgb(0,144,206)',
                                              "Mixed": 'rgb(211,211,211)'},
                          title='Gender distribution')
        	st.plotly_chart(fig2,use_container_width=True)

        df_map = pd.DataFrame()
        df_map['country'] = df_total['country_code'].value_counts().index
        df_map['counts'] = df_total['country_code'].value_counts().values
        data = dict(type='choropleth', 
                    locations=df_map['country'], 
                    z=df_map['counts'])

        layout = dict(title='Participating JJNOs', 
                      geo=dict(showframe=True,
                               projection={'type':'robinson'}))
        x = pg.Figure(data=[data], layout=layout)
        st.plotly_chart(x)



        df_age_dis = df_total[['name','age_division','cat_type','continent']].groupby(['cat_type','age_division','continent']).count().reset_index()
        fig3 = px.bar(df_age_dis, x="age_division", y= "name",
                      color="cat_type", color_discrete_map=COLOR_MAP,
                      text='name',hover_data=["continent"],
                      title="age_division and disciplines")
        st.plotly_chart(fig3)

        df_medal = df_ini[['country','rank','name']].groupby(['country','rank']).count().reset_index()
        fig4 = px.bar(df_medal[df_medal['rank']<4], x='country', y= 'name', color='rank',text='name', title="Medals")
        fig4.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig4)

    elif mode =='Single Event': 

        # for individual events
        df_evts_plot = df_ini[['id', 'name', 'cat_type', 'age_division']].groupby(['id', 'cat_type', 'age_division']).count().reset_index()
        df_evts_plot = df_evts_plot.join(df_evts[['id','startDate']].set_index('id'), on='id')
        fig3 = px.area(df_evts_plot, x="startDate", y='name', color="cat_type", color_discrete_map=COLOR_MAP, line_group="age_division")
        st.plotly_chart(fig3)



        df_medal = df_ini[['country','rank','name']].groupby(['country','rank']).count().reset_index()
        fig4 = px.bar(df_medal[df_medal['rank']<4], x='country', y= 'name', color='rank',text='name', title="Medals")
        fig4.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig4)

    else:

        with st.expander("Hide/include indivudual countries"):
            country_sel = df_ini['country'].unique().tolist()
            countryt_select = st.multiselect('Select the country',
                                             country_sel,
                                             country_sel)
            df_ini = df_ini[df_ini['country'].isin(countryt_select)]
