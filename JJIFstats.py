'''
Read in data from sportdata and old sources (json) and display
statistic on JJIF

Names are mapped using:
https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e
force A and B as a CSR matrix.


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
AGE_INP = ["U16", "U18", "U21", "Adults", "U14", "U12", "U10", "U15", "Master"]
# preselected age_divisions
AGE_SEL = ["U16", "U18", "U21", "Adults"]

# the supported disciplines
DIS_INP = ["Duo", "Show", "Jiu-Jitsu", "Fighting", "Contact"]
# preselected disciplines
DIS_SEL = ["Duo", "Show", "Jiu-Jitsu", "Fighting"]

# continents
CONT_INP = ["Europe", "Pan America", "Africa", "Asia", "Oceania"]

# types of events
EVENT_TYPE_INP = ['National Championship', 'Continental Championship',
                  'World Championship', 'A Class Tournament',
                  'B Class Tournament',
                  'World Games / Combat Games']
# preselected types of events
EVENT_TYPE_SEL = ['Continental Championship',
                  'World Championship', 'A Class Tournament',
                  'B Class Tournament',
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

COLOR_MAP_AGE = {"Adults": 'rgb(243, 28, 43)',
                 "U21": 'rgb(0,144,206)',
                 "U18": 'rgb(211,211,211)',
                 "U16": 'rgb(105,105,105)',
                 "U14": 'rgb(255,255,255)'}

COLOR_MAP_ETYPE = {"World Championship": 'rgb(243, 28, 43)',
                   "Continental Championship": 'rgb(0,144,206)',
                   "A Class Tournament": 'rgb(211,211,211)',
                   "B Class Tournament": 'rgb(105,105,105)',
                   "World Games / Combat Games": 'rgb(255,255,255)'}


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
     Reads in a csv  and convert category ids to category names

    '''
    inp_file = pd.read_csv('https://raw.githubusercontent.com/ClaudiaBehnke86/JJIFsupportFiles/main/catID_name.csv', sep=';')
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
                                   ('History', 'Single Event', 'Countries', 'World Games' , 'Events Overview'))
    dstart_in = st.sidebar.date_input("From", dt.date(2000, 1, 1))
    dend_in = st.sidebar.date_input("To", dt.date.today())

    age_select_in = st.sidebar.multiselect('Select the age divisions',
                                           AGE_INP,
                                           AGE_SEL)
    dis_select_in = st.sidebar.multiselect('Select the disciplines',
                                           DIS_INP,
                                           DIS_SEL)
    para_in = st.sidebar.selectbox('Para setting',
                                   ('Include', 'Exclude', 'Only'),
                                   help='Include = Include Para in statistic,\
                                   Exclude = Exclude Para in statistic , \
                                   Only = Shows only Para disciplines')
    cont_select_in = st.sidebar.multiselect('Select the continent',
                                            CONT_INP,
                                            CONT_INP)
    evtt_select_in = st.sidebar.multiselect('Select the event type',
                                            EVENT_TYPE_INP,
                                            EVENT_TYPE_SEL)

    return age_select_in, dis_select_in, cont_select_in, dstart_in, \
        dend_in, evtt_select_in, mode_in, para_in


def get_events(dstart_in, dend_in, evtt_select, user, password):
    '''
    takes parameters an makes APIcall

    Parameters
    ----------
    dstart_in
        start date (from here on events are called) [datetime.date]
    dend_in
        end date (till here events are called) [datetime.date]
    evtt_select
        type of event selected. See EVENT_TYPE_INP for options [list]
    user
        sportdata API user name
    password
        sportdata API password

    '''

    start_str = dstart.strftime("%Y%m%d")
    end_str = dend.strftime("%Y%m%d")
    uri = "https://www.sportdata.org/ju-jitsu/rest/events/ranked/" + \
        start_str + "/" + end_str + "/"

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)

    d_in = response.json()
    df2 = json_normalize(d_in)
    df2 = df2[['id', 'country_code', 'name', 'eventtype', 'startDate']]
    df2['startDate'] = df2['startDate'].str[:10]

    # one needs to get events which are not in the db
    # returns csv as df
    inp_file = pd.read_csv("Events.csv", sep=';')
    df_wg = inp_file[
        ['id', 'country_code', 'name', 'eventtype', 'startDate']
    ]

    df2 = pd.concat([df2, df_wg])
    df2['startDate'] = pd.to_datetime(df2["startDate"], format="%Y-%m-%d").dt.date

    df2 = df2[df2['startDate'].between(dstart_in, dend_in)]
    df2 = df2[df2['eventtype'].isin(evtt_select)]

    return df2


def update_events(df_evts_in, age_select_in, dis_select_in, cont_select_in, evtt_in):
    '''
    allows a selection of events that are analyzed

    Parameters
    ----------
    df_events_in
        data frame with all events [data frame]
    age_select_in
        age division selected. See AGE_INP for options [list]
    dis_select_in
        age division selected. See AGE_INP for options [list]
    user
        sportdata API user name
    password
        sportdata API password

    '''

    frames_merge = []
    if len(age_select_in) > 0 and len(dis_select_in) > 0 and len(cont_select_in) > 0 and len(evtt_in) > 0:

        my_bar = st.progress(0)
        # read in a all events and add to df
        with st.spinner('Recalculate events'):
            for count, val in enumerate(df_evts_in['id'].tolist()):
                if ds.check_cache(val):
                    df_file = ds.get_data(val)
                    if len(df_file) > 0:
                        frames_merge.append(df_file)
                else:
                    d_in = file_check(val, st.secrets['user'],
                                      st.secrets['password'],
                                      st.secrets['user1'],
                                      st.secrets['password1'],
                                      st.secrets['url'])
                    if len(d_in) > 0:
                        df_file = json_normalize(d_in['results'])
                        if len(df_file) > 0:
                            df_file = df_file[['category_id', 'category_name', 'country_code', 'rank', 'name']]
                            df_file['id'] = val
                            # add event dataframe to general data
                            ds.set_data(val, df_file)
                            frames_merge.append(df_file)
                    else:
                        ds.set_data(val, pd.DataFrame())
                my_bar.progress((count+1)/len(df_evts_in['id'].tolist()))
    return frames_merge


@st.cache_data
def file_check(numb, user, password, user1, password1, data_url):
    '''
    runs over all files in event list and get them from
    1. local database
    2. sportdata API or
    3. return that event does not exist

    Parameters
    ----------
    numb
        unique identifier [str]
    user
        sportdata API user name
    password
        sportdata API password
    user1
        local database user name
    password
        local database password
    data_url
        url of local database
    '''

    # 1. local database
    db_string = "curl -u " + user1 + ":" + password1 + \
        " " + data_url + numb + ".json > " + numb + ".json"
    os.system(db_string)
    d_in = {}

    if numb == "20000006":
        print("skip" + numb)
        d_empt = {}
        return d_empt
    if numb == "WCh2017":
        with open(numb+".json", "r", encoding="utf-8") as f_in:
            d_in = json.load(f_in)
            return d_in

    if os.stat(numb+".json").st_size > 256:
        with open(numb+".json", "r", encoding="utf-8") as f_in:
            d_in = json.load(f_in)
            return d_in

    # 2. sportdata API
    else:
        print("File " + numb + " does not appear to in local database. Try online")
        uri2 = 'https://www.sportdata.org/ju-jitsu/rest/event/resultscomplete/' + str(numb) + '/'
        response2 = requests.get(uri2,
                                 auth=HTTPBasicAuth(user, password),
                                 timeout=5)
        f_api = response2.json()

        # 3. event does not exist
        if len(f_api) <= 0:
            print("Event " + numb + " is not online, skip event")
            # return empty dataframe
            d_empt = {}
            return d_empt
        else:
            return f_api


def conv_to_type(df_in, type_name, type_list):
    '''
    checks strings in data frames and
    replaces them with types based on the _INP lists (see line 28 - 49)

    Parameters
    ----------
    df_in
        data frame to check [str]
    type_name
        the name of the _INP list to check [str]
    type_list
        of the _INP list to check [list]
    '''
    for inp_str in type_list:
        df_in[type_name].where(~(df_in[type_name].str.contains(inp_str)),
                               other=inp_str, inplace=True)

    return df_in[type_name]


def ngrams(string, n_gr=3):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    used to check for similar names
    Parameters
    ----------
    string
        input string
    n_gr
        ?

    '''
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams_in = zip(*[string[i:] for i in range(n_gr)])
    return [''.join(ngram_in) for ngram_in in ngrams_in]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    force A and B as a CSR matrix.
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
    return csr_matrix((data, indices, indptr), shape=(M, N))


def get_matches_df(sparse_matrix, name_vector, top=100):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    unpacks the resulting sparse matrix
    '''
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if sparsecols.size > top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity_in = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similarity_in[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similarity': similarity_in})


# Main program starts here
st.header('JJIF statistic')

with st.expander("Thanks to"):
    st.success('The Data contributors: \n Geert Assmann \n Nicolas \'Niggi\' Baez', icon="✅")

# create data store
ds = DataStore()

IOC_ISO = read_in_iso()
key_map = read_in_catkey()
age_select, dis_select, cont_select, dstart, dend, evtt, mode, para_inp = data_setting()


if mode == "World Games":
    evtt = ["World Games / Combat Games"]
    dstart = dt.date(1997, 1, 1)


df_evts = get_events(dstart, dend, evtt, st.secrets['user'], st.secrets['password'])
evt_sel = df_evts['name'].unique()

with st.expander("Details of events"):
    st.write(df_evts)

if mode == 'Single Event':
    evtt_select = st.selectbox("Select the event:",
                               evt_sel)
    if len(evtt_select) > 0:
        df_evts = df_evts[df_evts['name']==evtt_select]

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
    df_ini['name'].replace("Ž", "Z", regex=True, inplace=True)
    df_ini['name'].replace("Ń", "N", regex=True, inplace=True)
    df_ini['name'].replace(",", " ", regex=True, inplace=True)
    df_ini['name'].replace("  ", " ", regex=True, inplace=True)
    df_ini['rank'] = df_ini['rank'].astype(int)
    df_ini['category_id'] = df_ini['category_id'].astype(int)

    # remove all categories which are not in key map and convert to hr name
    with st.expander("Show unsupported categories", expanded=False):
        df_excluded = df_ini[~df_ini['category_id'].isin(key_map.keys())]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("FRIENDSHIP"))]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("TEAM"))]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("Team"))]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("R2"))]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("R5"))]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("Final"))]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("FINAL"))]
        df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("DEMONSTRATION"))]
        st.write(df_excluded)
        if len(df_excluded['category_id'].unique()) > 0:
            st.write("There are : " +
                     str(len(df_excluded['category_id'].unique()))
                     + " categories not included")
    df_ini = df_ini[df_ini['category_id'].isin(key_map.keys())]
    df_ini['category_name'] = df_ini['category_id'].replace(key_map)

    # merge similar names
    cat_list = df_ini['category_name'].unique().tolist()

    # loop over all categories
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

    with st.expander('Details on name matching', expanded=False):
        st.write('Similar names were matched to avoid double counting. This is based on:')
        st.write('https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e')
        similarity = st.number_input('similarity', min_value=0.4,
                                     max_value=0.9, value=0.6,
                                     help="small number means more matches, high number only exact matches"
                                     )
    # create empty temporary list for events to fix names
    list_df_new = []

    for i, val in enumerate(cat_list):
        df_new = df_ini[df_ini['category_name'].str.contains(str(val))]

        # re-index the names column to continuous index starting at
        names_types = pd.Series(df_new['name'].values)

        if len(names_types) > 1:
            tf_idf_matrix = vectorizer.fit_transform(names_types)
            if len(names_types) > 4:
                matches = awesome_cossim_top(tf_idf_matrix,
                                             tf_idf_matrix.transpose(),
                                             10, 0.4)
            else:
                matches = awesome_cossim_top(tf_idf_matrix,
                                             tf_idf_matrix.transpose(),
                                             4, 0.4)
            # store the  matches into new dataframe called matched_df
            matches_df = get_matches_df(matches, names_types, top=200)
            # For removing all exact matches
            matches_df = matches_df[matches_df['similarity'] < 0.99999]
            # create a mapping between names in form of a dict
            matches_df = matches_df[matches_df['similarity'] > similarity]
            dict_map = dict(zip(matches_df.left_side, matches_df.right_side))
            df_new.loc[:, 'name'] = df_new['name'].replace(dict_map)

            list_df_new.append(df_new)

            # if len(dict_map) > 0:
            #    print('fixing ' + str(len(dict_map)) + ' issues with names')

    # overwrite existing df_ini with events with name issues fixed
    df_ini = pd.concat(list_df_new)

    # convert neutral athletes into Liechtenstein
    # (make sure to change if we ever have a JJNO there)
    df_ini["country_code"].replace("JJIF", "LIE", regex=True, inplace=True)
    df_ini["country_code"].replace("JIF", "LIE", regex=True, inplace=True)
    df_ini["country_code"].replace("AIN", "LIE", regex=True, inplace=True)

    # replace wrong country codes and make all ISO
    df_ini["country_code"].replace("RJF", "RUS", regex=True, inplace=True)
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

    # String comparison does not handle + well... replaced with p in csv
    # and here replaced back
    df_ini['category_name'].replace(" p", " +", regex=True, inplace=True)

    df_ini['cat_type'] = df_ini['category_name']
    df_ini['cat_type'] = conv_to_type(df_ini, 'cat_type', DIS_INP)

    df_ini['age_division'] = df_ini['category_name']
    df_ini['age_division'] = conv_to_type(df_ini, 'age_division', AGE_INP)

    # remove what is not selected
    df_ini = df_ini[df_ini['cat_type'].isin(dis_select)]
    df_ini = df_ini[df_ini['age_division'].isin(age_select)]
    df_ini = df_ini[df_ini['continent'].isin(cont_select)]
    if para_inp == 'Exclude':
        df_ini = df_ini[~df_ini['category_name'].str.contains("Para")]
    elif para_inp == 'Only':
        df_ini = df_ini[df_ini['category_name'].str.contains("Para")]
    else:
        print("Include Para")

    if mode == "Countries":
        country_sel = df_ini['country'].unique().tolist()
        country_sel.sort()
        countryt_select = st.selectbox('Select the country',
                                       country_sel)
        if len(countryt_select) > 0:
            df_ini = df_ini[df_ini['country'] == countryt_select]

    df_par = df_ini.copy()
    df_par = df_par.join(df_evts[['id', 'startDate']].set_index('id'), on='id')

    df_min = df_par[['country', 'name', 'category_name', 'startDate']].groupby(['country', 'name', 'category_name']).min().reset_index()
    df_min.rename(columns={"startDate": "entryDate"}, inplace=True)

    df_max = df_par[['country', 'name', 'category_name', 'startDate']].groupby(['country', 'name', 'category_name']).max().reset_index()
    df_max.rename(columns={"startDate": "leavingDate"}, inplace=True)
    df_max['leavingDate'] = df_max['leavingDate'] + pd.offsets.DateOffset(years=2)
    df_total = pd.merge(df_min, df_max)

    df_total['long_id'] = df_total['country'] + "_" + df_total['name'] + "_" +\
        df_total['category_name']
    df_total['gender'] = df_total['category_name']
    df_total['gender'].where(~(df_total['gender'].str.contains("Men")),
                             other="Men", inplace=True)
    df_total['gender'].where(~(df_total['gender'].str.contains("Women")),
                             other="Women", inplace=True)
    df_total['gender'].where(~(df_total['gender'].str.contains("Mixed")),
                             other="Mixed", inplace=True)
    df_total['gender'].where(~(df_total['gender'].str.contains("Open")),
                             other="Open", inplace=True)

    # convert country names to codes and check continents
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

    # merge the entries to a time range
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

    # convert country names to codes and check continents
    df_time['country_code'] = df_time['country'].apply(lambda x: pc.country_name_to_country_alpha2(x))
    df_time['continent'] = df_time['country_code'].apply(lambda x: pc.country_alpha2_to_continent_code(x))
    df_time['continent'] = df_time['continent'].apply(lambda x: pc.convert_continent_code_to_continent_name(x))
    df_time['continent'].where(~(df_time['continent'].str.contains("South America")),
                               other="Pan America", inplace=True)
    df_time['continent'].where(~(df_time['continent'].str.contains("North America")),
                               other="Pan America", inplace=True)

    # add number of participants to df_evnet
    df_evt_part = df_ini[['id', 'name']].groupby(['id']).count().reset_index()
    df_evt_part = df_evt_part.rename(columns={'name': 'no participants'})
    df_evts = pd.merge(df_evts, df_evt_part, on='id', how='outer')

    # start graphics here
    if mode == 'Events Overview':

        df_evts['country_code'] = df_evts['country_code'].replace(IOC_ISO)
        df_evts['country_code'] = df_evts['country_code'].apply(lambda x: pc.country_alpha2_to_country_name(x))
        df_evts['country_code'] = df_evts['country_code'].apply(lambda x: pc.country_name_to_country_alpha3(x))

        st.write(df_evts)

        df_dip_evts = df_evts[['name', 'startDate', 'no participants','eventtype','country_code']]
        fig1 = px.bar(df_dip_evts, x='startDate', y='no participants', color='eventtype',
                       hover_data=['startDate', 'name', 'no participants','country_code'],
                       title="Events per year",
                       color_discrete_map=COLOR_MAP_ETYPE,
                       labels={
                                "dates": "Date [year]",
                                "name": "Name of event"
                                }
                       )
        fig1.update_layout(xaxis_range=[df_dip_evts['startDate'].min(), dend])
        fig1.update_traces(width=1204800000)
        st.plotly_chart(fig1)

        df_map1 = pd.DataFrame()
        df_map1['country'] = df_evts['country_code'].value_counts().index
        df_map1['counts'] = df_evts['country_code'].value_counts().values

        data = dict(type='choropleth',
                    locations=df_map1['country'], z=df_map1['counts'])

        layout = dict(title='Organised Events',
                      geo=dict(showframe=True,
                               projection={'type': 'robinson'}))
        x = pg.Figure(data=[data], layout=layout)
        x.update_geos(
                showcountries=True, countrycolor="black"
        )
        st.plotly_chart(x)

    elif mode == 'History':

        func_of = st.radio("Display time evolution for:",
                           ('Continent', 'Discipline', 'Age Divisions'),
                           horizontal=True)
        if func_of == 'Discipline':
            fuc_of_ty = 'cat_type'
            col_sel = COLOR_MAP
        elif func_of == 'Continent':
            fuc_of_ty = 'continent'
            col_sel = COLOR_MAP_CON
        else:
            fuc_of_ty = 'age_division'
            col_sel = COLOR_MAP_AGE

        df_timeev = df_time[['dates', 'name', fuc_of_ty]].groupby(['dates', fuc_of_ty]).count().reset_index()
        fig1 = px.area(df_timeev, x='dates', y='name', color=fuc_of_ty,
                       title="Time evolution of JJIF - Athletes (stacked)",
                       color_discrete_map=col_sel,
                       labels={
                                "dates": "Date [year]",
                                "name": "Number of Athletes"
                                }
                       )
        fig1.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig1)
        st.write("In total ", len(df_total), "Athletes")
        st.write("Currently", len(df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]), "Athletes active")
        fig1a = px.line(df_timeev, x='dates', y='name', color=fuc_of_ty,
                        title="Time evolution of JJIF - Athletes",
                        color_discrete_map=col_sel,
                        labels={
                                "dates": "Date [year]",
                                "name": "Number of Athletes"
                                }
                        )
        fig1a.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig1a)

        df_timeev_jjnos = df_time[['dates', 'country', fuc_of_ty]].groupby(['dates', fuc_of_ty]).nunique().reset_index()
        fig0 = px.area(df_timeev_jjnos, x='dates', y='country',
                       color=fuc_of_ty,
                       title="Time evolution of JJIF - JJNOs (stacked)",
                       color_discrete_map=col_sel,
                       labels={"dates": "Date [year]",
                               "name": "Number of JJNOs"}
                       )
        fig0.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.write("In total ", len(df_total['country'].unique()), "JJNOs")
        df_jjnocur = df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]
        st.write("Currently", len(df_jjnocur['country'].unique()), "JJNOs active")

        st.plotly_chart(fig0)

        df_timeev_jjnos_dis = df_time[['dates', 'country', fuc_of_ty]].groupby(['dates', fuc_of_ty]).nunique().reset_index()
        fig0a = px.line(df_timeev_jjnos_dis, x='dates', y='country',
                        color=fuc_of_ty,
                        title="Time evolution of JJIF - JJNOs",
                        color_discrete_map=col_sel,
                        labels={
                                "dates": "Date [year]",
                                "name": "Number of JJNOs"
                                }
                        )
        fig0a.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig0a)

        current_cat = st.checkbox('Show only currently active athletes',
                                  value=True)
        if current_cat:
            df_total = df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]

        df_cats = df_total[['name', 'category_name', 'cat_type', 'continent']].groupby(['category_name', 'cat_type', 'continent']).count().reset_index()
        fig_cats = px.bar(df_cats, x="category_name", y="name",
                          color="continent",
                          title="Athletes per category",
                          color_discrete_map=COLOR_MAP_CON,
                          labels={
                                "category_name": "Category",
                                "name": "Number of Athletes"
                                }
                          )
        fig_cats.update_layout(xaxis={'categoryorder': 'category ascending'})
        st.plotly_chart(fig_cats)

        with st.expander("Show numbers "):
            st.write(df_total[['name', 'category_name', 'cat_type']].groupby(['category_name', 'cat_type']).count().reset_index())

        df_cats_jjnos = df_total[['country', 'category_name', 'cat_type', 'continent']].groupby(['category_name', 'cat_type', 'continent']).nunique().reset_index()
        fig_cats_jjnos = px.bar(df_cats_jjnos, x="category_name", y="country",
                                color="continent", title="JJNOs per category",
                                color_discrete_map=COLOR_MAP_CON,
                                labels={
                                    "category_name": "Category",
                                    "country": "Number of JJNOs"
                                })
        fig_cats_jjnos.update_layout(xaxis={'categoryorder': 'category ascending'})
        st.plotly_chart(fig_cats_jjnos)

        with st.expander("Show numbers "):
            st.write(df_total[['country', 'category_name', 'cat_type']].groupby(['category_name', 'cat_type']).nunique().reset_index())

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
            st.plotly_chart(fig1, use_container_width=True)

        with right_column:
            df_gender = pd.DataFrame()
            df_gender['gender'] = df_total['gender'].value_counts().index
            df_gender['counts'] = df_total['gender'].value_counts().values
            fig2 = px.pie(df_gender, values='counts', names='gender',
                          color='gender',
                          color_discrete_map={"Women": 'rgb(243, 28, 43)',
                                              "Men": 'rgb(0,144,206)',
                                              "Mixed": 'rgb(211,211,211)',
                                              "Open": 'rgb(105,105,105)'},
                          title='Gender distribution')
            st.plotly_chart(fig2, use_container_width=True)

        df_map = pd.DataFrame()
        df_map['country'] = df_total['country_code'].value_counts().index
        df_map['counts'] = df_total['country_code'].value_counts().values
        data = dict(type='choropleth',
                    locations=df_map['country'], z=df_map['counts'])

        layout = dict(title='Participating JJNOs',
                      geo=dict(showframe=True,
                               projection={'type': 'robinson'}))
        x = pg.Figure(data=[data], layout=layout)
        st.plotly_chart(x)

        df_age_dis = df_total[['name', 'age_division', 'cat_type', 'continent']].groupby(['cat_type', 'age_division', 'continent']).count().reset_index()
        fig3 = px.bar(df_age_dis, x="age_division", y="name",
                      color="cat_type", color_discrete_map=COLOR_MAP,
                      text='name', hover_data=["continent"],
                      title="age_division and disciplines",
                      labels={
                            "age_division": "Age Division",
                            "name": "Number of Athletes",
                            "cat_type": "Discipline"
                             }
                      )
        st.plotly_chart(fig3)

        df_medal = df_ini[['country', 'rank', 'name']].groupby(['country', 'rank']).count().reset_index()
        # move Liechtenstein back to JJIF
        df_medal['country'].replace("Liechtenstein", "JJIF", regex=True, inplace=True)
        fig4 = px.bar(df_medal[df_medal['rank'] < 4], x='country', y='name',
                      color='rank', text='name', title="Medals",
                      labels={
                     "country": "Country code",
                     "name": "Number of Medals",
                     "rank": "Place"
                     })
        fig4.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig4)

    elif mode == 'Single Event':

        # for individual events
        df_cats_jjnos = df_total[['country', 'category_name', 'cat_type', 'continent']].groupby(['category_name', 'cat_type', 'continent']).nunique().reset_index()
        fig_cats_jjnos = px.bar(df_cats_jjnos, x="category_name", y="country",
                                color="continent", title="Athletes per category",
                                color_discrete_map=COLOR_MAP_CON,
                                labels={
                                        "category_name": "Category Name",
                                        "country": "Number of Athletes",
                                        "country": "Continent"})
        fig_cats_jjnos.update_layout(xaxis={'categoryorder': 'category ascending'})

        st.write("In total ", len(df_total['name'].unique()), "Athletes from",
                 len(df_total['country'].unique()), "JJNOs")
        st.plotly_chart(fig_cats_jjnos)

        df_numb_cat = df_total[['country', 'category_name']].groupby(['category_name']).nunique().reset_index()

        with st.expander("Show numbers"):
            df_numb_cat

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
            st.plotly_chart(fig1, use_container_width=True)

        with right_column:
            df_gender = pd.DataFrame()
            df_gender['gender'] = df_total['gender'].value_counts().index
            df_gender['counts'] = df_total['gender'].value_counts().values
            fig2 = px.pie(df_gender, values='counts', names='gender',
                          color='gender',
                          color_discrete_map={"Women": 'rgb(243, 28, 43)',
                                              "Men": 'rgb(0,144,206)',
                                              "Mixed": 'rgb(211,211,211)'},
                          title='Gender distribution')
            st.plotly_chart(fig2, use_container_width=True)

        df_medal = df_ini[['country', 'rank', 'name']].groupby(['country', 'rank']).count().reset_index()
        # move Liechtenstein back to JJIF
        df_medal['country'].replace("Liechtenstein", "JJIF", regex=True, inplace=True)
        fig4 = px.bar(df_medal[df_medal['rank'] < 4], x='country', y='name',
                      color='rank', text='name', title="Medals",
                      labels={
                     "country": "Country code",
                     "name": "Number of Medals",
                     "rank": "Place"
                     })
        fig4.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig4)
        st.write("In total ", len(df_medal['country'][df_medal['rank'] < 4].unique()), "JJNOs in medal tally")

    elif mode == 'World Games':

        func_of = st.radio("Display:",
                           ('Continent', 'Discipline'),
                           horizontal=True)
        if func_of == 'Discipline':
            fuc_of_ty = 'cat_type'
            col_sel = COLOR_MAP
        else:
            fuc_of_ty = 'continent'
            col_sel = COLOR_MAP_CON

        st.write("In total ", len(df_total['name'].unique()), "Athletes from",
                 len(df_total['country'].unique()), "JJNOs")
        df_evts_plot = df_ini[['id', 'name', fuc_of_ty]].groupby(['id', fuc_of_ty]).count().reset_index()
        df_evts_plot = df_evts_plot.join(df_evts[['id', 'startDate']].set_index('id'), on='id')
        fig3 = px.bar(df_evts_plot, x="startDate", y='name', color=fuc_of_ty,
                      color_discrete_map=col_sel,
                      labels={
                              "startDate": "Year of The World Games",
                              "name": "Number of Athletes"
                              })
        st.plotly_chart(fig3)

        df_evts_plot_JJNOs = df_ini[['id', 'country', fuc_of_ty]].groupby(['id', fuc_of_ty]).nunique().reset_index()
        df_evts_plot_JJNOs = df_evts_plot_JJNOs.join(df_evts[['id', 'startDate']].set_index('id'), on='id')
        fig5 = px.bar(df_evts_plot_JJNOs, x="startDate", y='country', color=fuc_of_ty,
                      color_discrete_map=col_sel,
                      labels={
                              "startDate": "Year of The World Games",
                              "country": "Number of JJNOs"
                              })
        st.plotly_chart(fig5)

        df_medal = df_ini[['country', 'rank', 'name']].groupby(['country', 'rank']).count().reset_index()
        # move Liechtenstein back to JJIF
        df_medal['country'].replace("Liechtenstein", "JJIF", regex=True, inplace=True)
        fig4 = px.bar(df_medal[df_medal['rank'] < 4], x='country', y='name',
                      color='rank', text='name', title="Medals",
                      labels={
                     "country": "Country",
                     "name": "Number of Medals",
                     "rank": "Place"
                     })
        fig4.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig4)
        st.write("In total ", len(df_medal['country'][df_medal['rank'] < 4].unique()), "JJNOs in medal tally")

        df_map = pd.DataFrame()
        df_map['country'] = df_total['country_code'].value_counts().index
        df_map['counts'] = df_total['country_code'].value_counts().values
        data = dict(type='choropleth',
                    locations=df_map['country'], z=df_map['counts'])

        layout = dict(title='Participating JJNOs',
                      geo=dict(showframe=True,
                               projection={'type': 'robinson'}))
        x = pg.Figure(data=[data], layout=layout)
        st.plotly_chart(x)

    else:

        left_column, right_column = st.columns(2)

        st.write("In total ", len(df_ini['name'].unique()), "Athletes from",
                 str(countryt_select))
        st.write("Currently", len(df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]), "Athletes active")

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
            st.plotly_chart(fig1, use_container_width=True)

        with right_column:
            df_gender = pd.DataFrame()
            df_gender['gender'] = df_total['gender'].value_counts().index
            df_gender['counts'] = df_total['gender'].value_counts().values
            fig2 = px.pie(df_gender, values='counts', names='gender',
                          color='gender',
                          color_discrete_map={"Women": 'rgb(243, 28, 43)',
                                              "Men": 'rgb(0,144,206)',
                                              "Mixed": 'rgb(211,211,211)'},
                          title='Gender distribution')
            st.plotly_chart(fig2, use_container_width=True)

        df_timeev = df_time[['dates', 'name', 'cat_type']].groupby(['dates', 'cat_type']).count().reset_index()
        fig1a = px.line(df_timeev, x='dates', y='name', color='cat_type',
                        title="Time evolution of " + str(countryt_select) + " - Disciplines",
                        color_discrete_map=COLOR_MAP,
                        labels={
                                "dates": "Date [year]",
                                "name": "Number of Athletes",
                                "cat_type": "Discipline"
                                }
                        )
        fig1a.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig1a)

        df_timeev_age_cat = df_time[['dates', 'name', 'age_division']].groupby(['dates', 'age_division']).count().reset_index()
        fig1b = px.line(df_timeev_age_cat, x='dates', y='name', color='age_division',
                        title="Time evolution of " + str(countryt_select) + " - Age Divisions",
                        color_discrete_map=COLOR_MAP_AGE,
                        labels={
                                "dates": "Date [year]",
                                "name": "Number of Athletes",
                                "age_division": "Age Division"
                                }
                        )
        fig1b.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
        st.plotly_chart(fig1b)

        inner_join = pd.merge(df_ini,
                              df_evts,
                              on='id',
                              how='inner')

        df_medal = inner_join[['name_y', 'rank', 'name_x']].groupby(['name_y', 'rank']).count().reset_index()
        # move Liechtenstein back to JJIF
        df_medal['country'].replace("Liechtenstein", "JJIF", regex=True, inplace=True)
        fig4 = px.bar(df_medal[df_medal['rank'] < 4], x='name_y', y='name_x',
                      color='rank', text='name_x', title="Medals in Events",
                      labels={
                                "name_x": "Event Name",
                                "name_y": "Number of Medals",
                                "rank": "Place"
                                }
                      )
        fig4.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig4)


st.sidebar.markdown('<a href="mailto:sportdirector@jjif.org">Contact for problems</a>', unsafe_allow_html=True)

LINK = '[Click here for the source code](https://github.com/ClaudiaBehnke86/JJIFseeding)'
st.markdown(LINK, unsafe_allow_html=True)
