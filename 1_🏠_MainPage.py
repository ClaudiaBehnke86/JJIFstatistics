'''
Read in data from sportdata and old sources (json) and display
statistic on JJIF

Names are mapped using:
https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e
force A and B as a CSR matrix.
With sportse_dot_topn updated https://github.com/ing-bank/sparse_dot_topn/tree/master


This is the Main Program which does all the calculations and creates the data frames.
The plots are stored in different pages for streamlit (see pages folder)
'''

import datetime as dt
import os
import os.path
import re

import pandas as pd

import streamlit as st
import pycountry_convert as pc
import numpy as np

# for the name matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# some style for the page
st.set_page_config(
    page_title="JJIF Statistics",
    page_icon="https://i0.wp.com/jjeu.eu/wp-content/uploads/2018/08/jjif-logo-170.png?fit=222%2C160&ssl=1",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:sportdirector@jjif.org',
        'Report a bug': "https://github.com/ClaudiaBehnke86/JJIFstatistics",
        'About': "# JJIF statistics."
    }
)


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://i0.wp.com/jjeu.eu/wp-content/uploads/2018/08/jjif-logo-170.png?fit=222%2C160&ssl=1);
                background-repeat: no-repeat;
                padding-top: 200px;
                background-position: 50px 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

# some dicts to map colors in plotly graphs
COLOR_MAP = {"Jiu-Jitsu": 'rgb(243, 28, 43)',# red
             "Fighting": 'rgb(0,144,206)',# blue
             "Duo": 'rgb(211,211,211)',
             "Show": 'rgb(105,105,105)',
             "Contact": 'rgb(255,255,255)'}
# save as session state to support multi page apps
st.session_state['COLOR_MAP'] = COLOR_MAP

# some dictionaries for JJIF Colors
COLOR_MAP_CON = {"Europe": 'rgb(243, 28, 43)',
                 "Asia": 'rgb(0,144,206)',
                 "Pan America": 'rgb(211,211,211)',
                 "Africa": 'rgb(105,105,105)',
                 "Oceania": 'rgb(255,255,255)'}
# save as session state to support multi page apps
st.session_state['COLOR_MAP_CON'] = COLOR_MAP_CON

COLOR_MAP_AGE = {"Adults": 'rgb(243, 28, 43)',
                 "U21": 'rgb(0,144,206)',
                 "U18": 'rgb(211,211,211)',
                 "U16": 'rgb(105,105,105)',
                 "U14": 'rgb(255,255,255)'}
# save as session state to support multi page apps
st.session_state['COLOR_MAP_AGE'] = COLOR_MAP_AGE

COLOR_MAP_ETYPE = {"World Championship": 'rgb(243, 28, 43)',
                   "Continental Championship": 'rgb(0,144,206)',
                   "A Class Tournament": 'rgb(211,211,211)',
                   "B Class Tournament": 'rgb(105,105,105)',
                   "World Games / Combat Games": 'rgb(255,255,255)'}
st.session_state['COLOR_MAP_ETYPE'] = COLOR_MAP_ETYPE


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
    # create tow columns
    col1, col2 = st.columns(2)

    with col1:
        # from this date on events are complete
        dstart_in = st.date_input("From", dt.date(2003, 1, 2))
        age_select_in = st.multiselect('Select the age divisions',
                                       AGE_INP, AGE_SEL)

    with col2:
        dend_in = st.date_input("To", dt.date.today())
        dis_select_in = st.multiselect('Select the disciplines',
                                       DIS_INP, DIS_SEL)
    cont_select_in = st.multiselect('Select the continent',
                                    CONT_INP, CONT_INP)

    evtt_select_in = st.multiselect('Select the event type',
                                    EVENT_TYPE_INP, EVENT_TYPE_SEL)

    para_in = st.selectbox('Inclusive/Para Ju-Jitsu setting',
                           ('Include', 'Exclude', 'Only'),
                           help='Include = Include Inclusive/Para in statistic,\
                           Exclude = Exclude Inclusive/Para in statistic , \
                           Only = Shows only Inclusive/Para disciplines')

    with st.expander("Acknowledgements"):
        st.success('Thanks to the Data contributors:  \n Geert Assmann  \n Nicolas \'Niggi\' Baez  \n The DJJV team \n SPortdata', icon="✅")

    return age_select_in, dis_select_in, cont_select_in, dstart_in, \
        dend_in, evtt_select_in, para_in


@st.cache_data
def read_in_ini_csv(user1, password1, data_url):
    ''' Read in the initial files
     Reads in a .csv  and the event .csv

    '''

    with st.spinner('Wait for it...'):
        db_string = "curl -u " + user1 + ":" + password1 + \
            " " + data_url + "ini.csv > " + "ini.csv"

        os.system(db_string)

        if os.stat('ini.csv').st_size > 256:
            with open("ini.csv", "r", encoding="utf-8") as f_in:
                df_ini = pd.read_csv(f_in)

        db_string1 = "curl -u " + user1 + ":" + password1 + \
            " " + data_url + "events.csv > " + "events.csv"

        os.system(db_string1)

        if os.stat('events.csv').st_size > 0:
            with open("events.csv", "r", encoding="utf-8") as f_in1:
                df_evts_ini = pd.read_csv(f_in1)

    return df_ini, df_evts_ini


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
st.title('JJIF statistic')
add_logo()
st.write('Welcome to the JJIF statistic app')

st.error('Please stay on this first page until you see \
    balloons :balloon:  \n to make sure the data is loaded (takes ~5 seconds)')

st.write('JJIF Statistics is a tool for reading,\
    processing, and displaying statistics related \
    to the Ju-Jitsu International Federation (JJIF) events.')

st.success(" :arrow_left: You can have a look at the \
    different overview on the left hand menu")
IOC_ISO = read_in_iso()
key_map = read_in_catkey()

st.divider()
with st.expander('Details on name matching', expanded=False):
    st.write('Names of athletes were matched on similarity, since data is not stored in uniform format')
    st.markdown('''This means that , `Pippi Långstrump`, `Pippi Langstrump`, \
        `P Långstrump` and `Långstrump Pippi` are most likely the same person \
        and will be merged in the ranking list''')
    st.write('This is based on:')
    st.write('https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e')
    st.write('You can change the matching here:')
    similarity = st.number_input('Similarity', min_value=0.4,
                                 max_value=0.9, value=0.6,
                                 help="small number means more matches, high number only exact matches"
                                 )
st.divider()
st.write('Here you can include/exclude types of data from the overviews')
st.warning('If you are not sure, just leave the default values', icon='✅')
age_select, dis_select, cont_select, dstart, dend, evtt, para_inp = data_setting()
if dstart < dt.date(2003, 1, 1):
    st.warning("Old data only contains medals and not total results",  icon="⚠️")

# Check if you've already initialized the data
if 'dstart' not in st.session_state:
    # Save the data to session state
    st.session_state.dstart = dstart
# save as session state to support multi page apps
st.session_state['dstart'] = dstart

if 'dend' not in st.session_state:
    # Save the data to session state
    st.session_state.dend = dend
# save as session state to support multi page apps
st.session_state['dend'] = dend

df_ini, df_evts_ini = read_in_ini_csv(st.secrets['user1'], st.secrets['password1'], st.secrets['url'])

# make a copy to avoid evts_ini is used in session state
df_evts = df_evts_ini

# make sure that start date is converted back to a date
df_evts['startDate'] = pd.to_datetime(df_evts['startDate'])

# rename name columns for events to avoid mismatch with ini names
df_evts = df_evts.rename(columns={"name": "Name Event"})

# convert IOC codes to ISO codes using a dict
df_evts['country_code'] = df_evts['country_code'].replace(IOC_ISO)
df_evts['country'] = df_evts['country_code'].apply(
            lambda x: pc.country_alpha2_to_country_name(x))
df_evts['country_code'] = df_evts['country_code'].apply(lambda x: pc.country_alpha2_to_country_name(x))
df_evts['country_code'] = df_evts['country_code'].apply(lambda x: pc.country_name_to_country_alpha3(x))

# cleanup of ini df
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

# covert to correct data type
df_ini['rank'] = df_ini['rank'].astype(int)
df_ini['category_id'] = df_ini['category_id'].astype(int)

# remove all categories which are not in key map and convert to hr name
df_excluded = df_ini[~df_ini['category_id'].isin(key_map.keys())]

# remove residual categories from Show, Duo and Team competitions
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("FRIENDSHIP"))]
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("TEAM"))]
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("Team"))]
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("R2"))]
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("R5"))]
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("Final"))]
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("FINAL"))]
df_excluded = df_excluded[~(df_excluded['category_name'].str.contains("DEMONSTRATION"))]

# show if categories are not in system
# those can be added to
if len(df_excluded) > 0:
    with st.expander("Show unsupported categories", expanded=False):
        st.write(df_excluded)
        st.write("There are : " +
                 str(len(df_excluded['category_id'].unique()))
                 + " categories not included")

df_ini = df_ini[df_ini['category_id'].isin(key_map.keys())]
df_ini['category_name'] = df_ini['category_id'].replace(key_map)

# merge identical category names
cat_list = df_ini['category_name'].unique().tolist()

# loop over all categories
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

# create empty temporary list to fix names
list_df_new = []

for i, val in enumerate(cat_list):
    df_new = df_ini[df_ini['category_name'].str.contains(str(val))]

    # re-index the names column to continuous index starting at
    names_types = pd.Series(df_new['name'].values)

    if len(names_types) > 1:
        tf_idf_matrix = vectorizer.fit_transform(names_types)
        # make sure it works with small categories
        if len(names_types) > 4:
            matches = sp_matmul_topn(tf_idf_matrix,
                                     tf_idf_matrix.transpose(),
                                     top_n=10, threshold=0.4, sort=True)

        else:
            matches = sp_matmul_topn(tf_idf_matrix,
                                     tf_idf_matrix.transpose(),
                                     top_n=4, threshold=0.4, sort=True)

        # store the  matches into new dataframe called matched_df
        matches_df = get_matches_df(matches, names_types, top=200)
        # For removing all exact matches
        matches_df = matches_df[matches_df['similarity'] < 0.99999]
        # create a mapping between names in form of a dict
        matches_df = matches_df[matches_df['similarity'] > similarity]
        dict_map = dict(zip(matches_df.left_side, matches_df.right_side))
        df_new.loc[:, 'name'] = df_new['name'].replace(dict_map)

        list_df_new.append(df_new)


# overwrite existing df_ini with events with name issues fixed
df_ini = pd.concat(list_df_new)

# convert neutral athletes into Liechtenstein
# (make sure to change if we ever have a JJNO there)
df_ini["country_code"].replace("JJIF", "LIE", regex=True, inplace=True)
df_ini["country_code"].replace("JIF", "LIE", regex=True, inplace=True)
df_ini["country_code"].replace("AIN", "LIE", regex=True, inplace=True)

# replace wrong country codes in data
df_ini["country_code"].replace("RJF", "RUS", regex=True, inplace=True)
df_ini["country_code"].replace("ENG", "GBR", regex=True, inplace=True)

# convert IOC codes to ISO codes using a dict
df_ini['country_code'] = df_ini['country_code'].replace(IOC_ISO)

# set the continent
df_ini['continent'] = df_ini['country_code'].apply(
            lambda x: pc.country_alpha2_to_continent_code(x))

# get ISO country name
df_ini['country'] = df_ini['country_code'].apply(
            lambda x: pc.country_alpha2_to_country_name(x))
df_ini['country_code'] = df_ini['country_code'].apply(
            lambda x: pc.country_alpha2_to_country_name(x))
df_ini['country_code'] = df_ini['country_code'].apply(
            lambda x: pc.country_name_to_country_alpha3(x))
df_ini['continent'] = df_ini['continent'].apply(
            lambda x: pc.convert_continent_code_to_continent_name(x))


# some JJIF adaptions
# we have a Pan American Union and not North and South Amerixa
df_ini['continent'].where(~(df_ini['continent'].str.contains("South America")),
                          other="Pan America", inplace=True)
df_ini['continent'].where(~(df_ini['continent'].str.contains("North America")),
                          other="Pan America", inplace=True)
# ISR is part of the European Union
df_ini['continent'].where(~(df_ini['country_code'].str.contains("ISR")),
                          other="Europe", inplace=True)
# TUR is part of the European Union
df_ini['continent'].where(~(df_ini['country_code'].str.contains("TUR")),
                          other="Europe", inplace=True)

# String comparison does not handle "+"" well... replaced with p in .csv
# and here replaced back
df_ini['category_name'].replace(" p", " +", regex=True, inplace=True)


# set all different categories in discipline, age division and gender
df_ini['cat_type'] = df_ini['category_name']
df_ini['cat_type'] = conv_to_type(df_ini, 'cat_type', DIS_INP)

df_ini['age_division'] = df_ini['category_name']
df_ini['age_division'] = conv_to_type(df_ini, 'age_division', AGE_INP)

df_ini['gender'] = df_ini['category_name']
df_ini['gender'].where(~(df_ini['gender'].str.contains("Men")),
                       other="Men", inplace=True)
df_ini['gender'].where(~(df_ini['gender'].str.contains("Women")),
                       other="Women", inplace=True)
df_ini['gender'].where(~(df_ini['gender'].str.contains("Mixed")),
                       other="Mixed", inplace=True)
df_ini['gender'].where(~(df_ini['gender'].str.contains("Open")),
                       other="Open", inplace=True)
# at the moment (6/2024) inclusive self defense is gender open
df_ini['gender'].where(~(df_ini['gender'].str.contains("Inclusive")),
                       other="Open", inplace=True)
# in some kids categories gendered were mixed
df_ini['gender'].where(~(df_ini['gender'].str.contains("Mix ")),
                       other="Open", inplace=True)

# remove what is not selected from user
df_ini = df_ini[df_ini['cat_type'].isin(dis_select)]
df_ini = df_ini[df_ini['age_division'].isin(age_select)]
df_ini = df_ini[df_ini['continent'].isin(cont_select)]

# Depending on user settings display Para categories
if para_inp == 'Exclude':
    df_ini = df_ini[~df_ini['category_name'].str.contains("Para")]
elif para_inp == 'Only':
    df_ini = df_ini[df_ini['category_name'].str.contains("Para")]
else:
    print("Include Para")

# merge the events and start dates
df_par = df_ini.copy()
df_par = df_par.join(df_evts[['id', 'startDate']].set_index('id'), on='id')

# find the date the athlete first entered
df_min = df_par[['country', 'name', 'category_name', 'startDate',
                 'gender', 'cat_type', 'age_division', 'continent',
                 'country_code']].groupby(['country', 'name',
                                           'category_name', 'gender',
                                           'cat_type', 'age_division',
                                           'continent', 'country_code'
                                           ]).min().reset_index()
df_min.rename(columns={"startDate": "entryDate"}, inplace=True)

# find the date the athlete last entered
df_max = df_par[['country', 'name', 'category_name', 'startDate',
                 'gender', 'cat_type', 'age_division', 'continent',
                 'country_code']].groupby(['country', 'name',
                                           'category_name', 'gender',
                                           'cat_type', 'age_division',
                                           'continent', 'country_code'
                                           ]).max().reset_index()
df_max.rename(columns={"startDate": "leavingDate"}, inplace=True)

# add two years to leaving date (That's when the ranking expires)
df_max['leavingDate'] = df_max['leavingDate'] + pd.offsets.DateOffset(years=2)
df_total = pd.merge(df_min, df_max)

# convert all into monthly rolling number to remove spikes
# create list with monthly steps
list_month = pd.date_range(start=dstart, end=dend, freq='MS').tolist()

# create empty temporary list
list_df_new = []
# adding column name to the respective columns
for i, val in enumerate(list_month[:-1]):
    start_date = pd.Timestamp(val)
    end_date = pd.Timestamp(list_month[i+1])

    # make a new df with all names which have start date and end data
    df_month = df_total[(df_total['entryDate'] <= start_date) & (df_total['leavingDate'] >= end_date)]
    df_month['month'] = val

    list_df_new.append(df_month)
df_time_smooth = pd.concat(list_df_new)

# add number of participants to df_event
df_evt_part = df_ini[['id', 'name']].groupby(['id']).count().reset_index()
df_evt_part = df_evt_part.rename(columns={'name': 'Number of Participants'})
df_evts = pd.merge(df_evts, df_evt_part, on='id', how='outer')

# Check if you've already initialized the data & store files in session state
if 'df_evts' not in st.session_state:
    # df that contains data of the events
    st.session_state.df_evts = df_evts

if 'df_total' not in st.session_state:
    # All events data (event & results) in one clean overview
    st.session_state.df_total = df_total

if 'df_par' not in st.session_state:
    # all athletes ever in ranking list (with entry and leaving date)
    st.session_state.df_par = df_par

if 'df_time_smooth' not in st.session_state:
    # df_par converted in to monthly times steps
    st.session_state.df_time_smooth = df_time_smooth

st.session_state['df_time_smooth'] = df_time_smooth
st.session_state['df_evts'] = df_evts
st.session_state['df_par'] = df_par
st.session_state['df_total'] = df_total

# makes sure user stays on main page until all data is loaded
st.balloons()

st.sidebar.markdown('<a href="mailto:sportdirector@jjif.org">Contact for problems</a>', unsafe_allow_html=True)

LINK = '[Click here for the source code](https://github.com/ClaudiaBehnke86/JJIFseeding)'
st.markdown(LINK, unsafe_allow_html=True)
