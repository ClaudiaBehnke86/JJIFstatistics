'''
 this page shows the historical development of JJIF

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


st.title("Time evolution of JJIF")
add_logo()

COLOR_MAP = st.session_state['COLOR_MAP']
COLOR_MAP_CON = st.session_state['COLOR_MAP_CON']
COLOR_MAP_AGE = st.session_state['COLOR_MAP_AGE']
dend = st.session_state['dend']
dstart = st.session_state['dstart']

# Retrieve the data from session state (and )
if 'df_time_smooth' in st.session_state:
    df_time_smooth = st.session_state['df_time_smooth']
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


if dstart < dt.date(2003, 1, 1):
    st.warning("Old data only contains medals and not total results",  icon="⚠️")

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

# static time evolution
df_timeev = df_time_smooth[['month', 'name', fuc_of_ty]].groupby(['month', fuc_of_ty]).count().reset_index()
fig1 = px.area(df_timeev, x='month', y='name', color=fuc_of_ty,
               title="Number of Athletes in Ranking list (stacked)",
               color_discrete_map=col_sel,
               labels={
                        "month": "Date [year]",
                        "name": "Number of Athletes"
                        }
               )
fig1.update_layout(xaxis_range=[dstart, dend])
st.plotly_chart(fig1)

st.write("In total ", len(df_total), "Athletes")
st.write("Currently", len(df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]), "Athletes active")
fig1a = px.line(df_timeev, x='month', y='name', color=fuc_of_ty,
                title="Number of Athletes in Ranking",
                color_discrete_map=col_sel,
                labels={
                        "month": "Date [year]",
                        "name": "Number of Athletes"
                        }
                )
fig1a.update_layout(xaxis_range=[dstart, dend])
st.plotly_chart(fig1a)

df_timeev_jjnos = df_time_smooth[['month', 'country', fuc_of_ty]].groupby(['month', fuc_of_ty]).nunique().reset_index()
fig0 = px.area(df_timeev_jjnos, x='month', y='country',
               color=fuc_of_ty,
               title="Number of JJNOs in Ranking (stacked)",
               color_discrete_map=col_sel,
               labels={"month": "Date [year]",
                       "name": "Number of JJNOs"}
               )
fig0.update_layout(xaxis_range=[dstart, dend])
st.write("In total ", len(df_total['country'].unique()), "JJNOs")
df_jjnocur = df_total[df_total['leavingDate'] > pd.Timestamp(dt.date.today())]
st.write("Currently", len(df_jjnocur['country'].unique()), "JJNOs active")

st.plotly_chart(fig0)

df_timeev_jjnos_dis = df_time_smooth[['month', 'country', fuc_of_ty]].groupby(['month', fuc_of_ty]).nunique().reset_index()
fig0a = px.line(df_timeev_jjnos_dis, x='month', y='country',
                color=fuc_of_ty,
                title="Number of JJNOs in Ranking",
                color_discrete_map=col_sel,
                labels={
                        "month": "Date [year]",
                        "name": "Number of JJNOs"
                        }
                )
fig0a.update_layout(xaxis_range=[dstart, dend])
st.plotly_chart(fig0a)

df_medal = df_par[['country', 'rank', 'name']].groupby(['country', 'rank']).count().reset_index()
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