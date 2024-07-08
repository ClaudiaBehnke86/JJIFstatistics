'''
Shows the developments of one JJNO

'''

import datetime as dt
import streamlit as st
import plotly.express as px
import pandas as pd


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


st.title("Details for one country")
add_logo()


COLOR_MAP = st.session_state['COLOR_MAP']
COLOR_MAP_AGE = st.session_state['COLOR_MAP_AGE']
df_time_smooth = st.session_state['df_time_smooth']
dstart = st.session_state['dstart']
dend = st.session_state['dend']

# Retrieve the data from session state (and )
if 'df_evts' in st.session_state:
    df_evts = st.session_state['df_evts']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")

if 'df_total' in st.session_state:
    df_total = st.session_state['df_total']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")

if 'df_par' in st.session_state:
    df_par = st.session_state['df_par']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")

# if only one country is selected remove all the others
country_sel = df_total['country'].unique().tolist()
country_sel.sort()
countryt_select = st.selectbox('Select the country',
                               country_sel,
                               help="Enter the country name (in English)")
df_total = df_total[df_total['country'] == countryt_select]
df_par = df_par[df_par['country'] == countryt_select]
df_time_smooth = df_time_smooth[df_time_smooth['country'] == countryt_select]

st.write("In total ", len(df_total['name'].unique()), "Athletes from",
         str(countryt_select))
st.write("Currently", len(df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]), "Athletes active")

left_column, right_column = st.columns(2)
with left_column:
    df_cat = pd.DataFrame()
    df_cat['cat_type'] = df_total['cat_type'].value_counts().index
    df_cat['counts'] = df_total['cat_type'].value_counts().values
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

df_timeev = df_time_smooth[['month', 'name', 'cat_type']].groupby(['month', 'cat_type']).count().reset_index()
fig1a = px.line(df_timeev, x='month', y='name', color='cat_type',
                title="Time evolution of " + str(countryt_select) + " - Disciplines",
                color_discrete_map=COLOR_MAP,
                labels={
                        "dates": "Date [year]",
                        "name": "Number of Athletes",
                        "cat_type": "Discipline"
                        }
                )
fig1a.update_layout(xaxis_range=[dstart, dend])
st.plotly_chart(fig1a)

df_timeev_age_cat = df_time_smooth[['month', 'name', 'age_division']].groupby(['month', 'age_division']).count().reset_index()
fig1b = px.line(df_timeev_age_cat, x='month', y='name', color='age_division',
                title="Time evolution of " + str(countryt_select) + " - Age Divisions",
                color_discrete_map=COLOR_MAP_AGE,
                labels={
                        "dates": "Date [year]",
                        "name": "Number of Athletes",
                        "age_division": "Age Division"
                        }
                )
fig1b.update_layout(xaxis_range=[dstart, dend])
st.plotly_chart(fig1b)

inner_join = pd.merge(df_par,
                      df_evts,
                      on='id',
                      how='inner')

df_medal = inner_join[['name', 'rank', 'Name Event']].groupby(['Name Event', 'rank']).count().reset_index()

fig4 = px.bar(df_medal[df_medal['rank'] < 4], x='Name Event', y='name',
              color='rank', text='Name Event', title="Medals in Events",
              labels={
                        "name Event": "Event Name",
                        "name": "Number of Medals",
                        "rank": "Place"
                        }
              )
fig4.update_xaxes(categoryorder='total descending')
st.plotly_chart(fig4)