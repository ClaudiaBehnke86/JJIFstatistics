'''
Read in data from sportdata and old sources (json) create ini cvs and events.csv
'''


from datetime import date
import json
import os
import requests

import streamlit as st

from requests.auth import HTTPBasicAuth
import pandas as pd
from pandas import json_normalize


def get_events(user, password):
    '''
    takes parameters an makes APIcall

    Parameters
    ----------
    user
        sportdata API user name
    password
        sportdata API password

    '''
    start_str = "20160101"
    end_str = date.today()

    uri = "https://www.sportdata.org/ju-jitsu/rest/events/ranked/" + \
        start_str + "/" + end_str.strftime("%Y%m%d") + "/"

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)

    d_in = response.json()
    df2 = json_normalize(d_in)
    df2 = df2.astype(str)
    df2 = df2[['id', 'country_code', 'name', 'eventtype', 'startDate']]
    df2['startDate'] = df2['startDate'].str[:10]

    # one needs to get events which are not in the db but curated by hand
    # returns csv as df
    with open("Events_added_by_hand.csv", "r", encoding="utf-8") as f_in:
                df_hc = pd.read_csv(f_in)

    df2 = pd.concat([df2, df_hc])
    df2['startDate'] = pd.to_datetime(df2["startDate"], format="%Y-%m-%d").dt.date

    return df2


def update_events(df_evts_in):
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
    # read in a all events and add to df
    for count, val in enumerate(df_evts_in['id'].tolist()):

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
                frames_merge.append(df_file)

    return frames_merge


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


# run code here:
df_evts = get_events(st.secrets['user'], st.secrets['password'])
st.write(df_evts)

df_evts.to_csv('events.csv', index=False)

frames = update_events(df_evts)
df_ini = pd.concat(frames)

st.write(df_ini)
df_ini.to_csv('ini.csv', index=False)


