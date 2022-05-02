'''
Read in data (json) and display statsitc on JJIF
'''

import json
import requests
import datetime as dt
from requests.auth import HTTPBasicAuth
import pandas as pd 
from pandas import json_normalize
import plotly.express as px
import streamlit as st 
import plotly.graph_objects as go
import pycountry_convert as pc

import plotly.graph_objs as pg
import numpy as np
import os

# the supported age_divisions
AGE_INP = ["U16", "U18", "U21", "Adults", "U14", "U12", "U10", "U15"]
AGE_SEL = ["U16", "U18", "U21", "Adults"]  # preselected age_divisions

# the supportes disciplines
DIS_INP = ["Duo", "Show", "Jiu-Jitsu", "Fighting", "Contact", "Para"]
DIS_SEL = ["Duo", "Show", "Jiu-Jitsu", "Fighting", "Contact"]  # presel discip

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
    IOC_ISO = inp_file[
        ['NOC','ISO code']
    ].set_index('NOC').to_dict()['ISO code']

    return IOC_ISO


def read_in_catkey():
    ''' Read in file
     - HELPER FUNCTION
     Reads in a csv  and convert catergory ids to catergoy names

    '''
    inp_file = pd.read_csv("catID_name.csv", sep=';')
    key_map = inp_file[
        ['cat_id', 'name']
    ].set_index('cat_id').to_dict()['name']

    return key_map


def data_setting():
    ''' The sidebar elements for the selection
    '''
    dstart = st.sidebar.date_input("From", dt.date(2010, 1, 1))
    dend = st.sidebar.date_input("To", dt.date.today())

    age_select = st.sidebar.multiselect('Select the age divisions',
                                        AGE_INP,
                                        AGE_SEL)
    dis_select = st.sidebar.multiselect('Select the disciplines',
                                        DIS_INP,
                                        DIS_SEL)
    cont_select = st.sidebar.multiselect('Select the continent',
                                         CONT_INP,
                                         CONT_INP)
    evtt_select = st.sidebar.multiselect('Select the event type',
                                         EVENT_TYPE,
                                         EVENT_INP)

    return age_select, dis_select, cont_select, dstart, dend, evtt_select


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
        with st.spinner('Recalculate events'):
            for i, val in enumerate(events):
                d = FileCheck(val, st.secrets['user'], st.secrets['password'], st.secrets['user1'],
                     st.secrets['password1'], st.secrets['url'])
                if len(d) > 0:
                    df_file = json_normalize(d['results'])
                    df_file = df_file[['category_id', 'category_name', 'country_code', 'rank', 'name']]
                    df_file['id'] = val
                    frames.append(df_file)

    return frames


def FileCheck(numb, user, password, user1, password1, data_url):
    '''
    runs over all files in event list and get them from
    sportdata or from the local database

    '''

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
    replaces them with defaul input
    '''
    for x in type_list:
        df[type_name].where(~(df[type_name].str.contains(x)),
                            other=x, inplace=True)
    return df[type_name]

IOC_ISO = read_in_iso()
key_map = read_in_catkey()
age_select, dis_select, cont_select, dstart, dend, evtt = data_setting()

checkfile = st.checkbox("Get new data", value=True)
if checkfile:
    df_evts = get_events(dstart, dend, evtt, st.secrets['user'], st.secrets['password'])
    # create a datastore
    # SINGLETON (design pattern)
    # ds = DataStore()
    # test if frames are already in datastore
    # if not ds.are_frame_already_loaded():
    #   frames = update_events ...
    #   ds.add_frames(frames)
    #   ds.add_flag_frames_are_loaded()
    # else
    #    frames = ds.frames
frames = update_events(df_evts, age_select, dis_select, cont_select, evtt)


if len(frames) == 0:
    st.write("please select at least one item in each category")
else:
    # cleanup of df
    df_ini = pd.concat(frames)
    df_ini['name'] = df_ini['name'].apply(lambda x: x.upper())
    df_ini['name'].replace("  ", " ", regex=True, inplace=True)
    # remove different characters in names
    df_ini['name'].replace("-", "/", regex=True, inplace=True)
    df_ini['name'].replace(" / ", "/", regex=True, inplace=True)
    df_ini['name'].replace(" /", "/", regex=True, inplace=True)
    df_ini['name'].replace("/ ", "/", regex=True, inplace=True)
    df_ini['name'].replace("Ö", "OE", regex=True, inplace=True)
    df_ini['name'].replace("Ä", "AE", regex=True, inplace=True)
    df_ini['name'].replace("Ć", "C", regex=True, inplace=True)

    df_ini['rank'] = df_ini['rank'].astype(int)
    df_ini['category_id'] = df_ini['category_id'].astype(int)
    # remove all categories which are not in key map and convert to hr name
    with st.expander("Show unsupported categories"):
        st.write(df_ini[~df_ini['category_id'].isin(key_map.keys())])
    df_ini = df_ini[df_ini['category_id'].isin(key_map.keys())]
    df_ini['category_name'] = df_ini['category_id'].replace(key_map)

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
    df_ini['cat_type'] = df_ini['category_name']
    df_ini['cat_type'] = conv_to_type(df_ini, 'cat_type', DIS_INP)

    df_ini['age_division'] = df_ini['category_name']
    df_ini['age_division'] = conv_to_type(df_ini, 'age_division', AGE_INP)

    # remove what is not selected
    df_ini = df_ini[df_ini['cat_type'].isin(dis_select)]
    df_ini = df_ini[df_ini['age_division'].isin(age_select)]
    df_ini = df_ini[df_ini['continent'].isin(cont_select)]

    with st.expander("Hide/include indivudual countries"):
        country_sel = df_ini['country'].unique().tolist()
        countryt_select = st.multiselect('Select the country',
                                         country_sel,
                                         country_sel)
        df_ini = df_ini[df_ini['country'].isin(countryt_select)]


    df_par = df_ini.copy()
    df_par = df_par.join(df_evts[['id','startDate']].set_index('id'), on='id') 
    df_min = df_par[['country', 'name', 'category_name', 'startDate']].groupby(['country', 'name', 'category_name']).min().reset_index()
    df_min.rename(columns={"startDate": "entryDate"}, inplace=True)
    df_max = df_par[['country', 'name', 'category_name', 'startDate']].groupby(['country', 'name', 'category_name']).max().reset_index()
    df_max.rename(columns={"startDate": "leavingDate"}, inplace=True)
    df_max['leavingDate'] = df_max['leavingDate'] + pd.offsets.DateOffset(years=2)
    df_total = pd.merge(df_min, df_max)


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


    df_timeev = df_time[['dates', 'name', 'cat_type']].groupby(['dates', 'cat_type']).count().reset_index()
    fig1 = px.area(df_timeev, x='dates', y='name', color='cat_type', title="Time evolution of JJIF - Athletes",
                  color_discrete_map=COLOR_MAP)
    fig1.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
    st.plotly_chart(fig1)

    df_timeev_jjnos = df_time[['dates', 'country', 'continent']].groupby(['dates', 'continent']).nunique().reset_index()
    fig0 = px.area(df_timeev_jjnos, x='dates', y='country', color='continent', title="Time evolution of JJIF - JJNOs",
                  color_discrete_map=COLOR_MAP_CON)
    fig0.update_layout(xaxis_range=[df_total['entryDate'].min(), dend])
    st.plotly_chart(fig0)


  #  df_cats = df_total[['name','category_name','age_division','cat_type','continent']].groupby(['cat_type','age_division','continent']).count().reset_index()
  
   # fig_cats = px.bar(df_cats, x="category_name", y="count", color="medal", title="Long-Form Input")
   # fig_cats.show()


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

    # for individual events
    df_evts_plot = df_ini[['id', 'name', 'cat_type', 'age_division']].groupby(['id', 'cat_type', 'age_division']).count().reset_index()
    df_evts_plot = df_evts_plot.join(df_evts[['id','startDate']].set_index('id'), on='id')
    fig3 = px.area(df_evts_plot, x="startDate", y='name', color="cat_type", color_discrete_map=COLOR_MAP, line_group="age_division")
    st.plotly_chart(fig3)

    df_medal = df_ini[['country','rank','name']].groupby(['country','rank']).count().reset_index()
    fig4 = px.bar(df_medal[df_medal['rank']<4], x='country', y= 'name', color='rank',text='name', title="Medals")
    fig4.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig4)