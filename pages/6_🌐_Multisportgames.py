'''
 this page shows multi sport events

'''

import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objs as pg


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


st.title("Multisport Events")
add_logo()

# Retrieve the data from session state (and )
if 'df_evts' in st.session_state:
    df_evts = st.session_state['df_evts']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")

if 'df_par' in st.session_state:
    df_par = st.session_state['df_par']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")

COLOR_MAP = st.session_state['COLOR_MAP']
COLOR_MAP_CON = st.session_state['COLOR_MAP_CON']

# Only select MS games
evtt = "World Games / Combat Games"
ms_list = df_evts['id'][df_evts['eventtype'] == evtt].unique().tolist()

df_ms = df_par[df_par['id'].isin(ms_list)]

# show events on screen
with st.expander("Details of events"):
    st.write(df_evts[df_evts['eventtype'] == evtt])

# show the data as a function of discipline or continent
func_of = st.radio("Display:",
                   ('Continent', 'Discipline'),
                   horizontal=True)
if func_of == 'Discipline':
    fuc_of_ty = 'cat_type'
    col_sel = COLOR_MAP
else:
    fuc_of_ty = 'continent'
    col_sel = COLOR_MAP_CON

st.write("In total ", len(df_ms['name'].unique()), "Athletes from",
         len(df_ms['country'].unique()), "JJNOs")

df_evts_plot = df_ms[['id', 'name', fuc_of_ty]].groupby(['id', fuc_of_ty]).count().reset_index()
df_evts_plot = df_evts_plot.join(df_evts[['id', 'startDate', 'Name Event']].set_index('id'), on='id')
fig3 = px.bar(df_evts_plot, x="startDate", y='name', color=fuc_of_ty,
              color_discrete_map=col_sel, text='Name Event',
              labels={
                      "startDate": "Year of the Games",
                      "name": "Number of Athletes"
                      })
st.plotly_chart(fig3)

df_evts_plot_JJNOs = df_ms[['id', 'country', fuc_of_ty]].groupby(['id', fuc_of_ty]).nunique().reset_index()
df_evts_plot_JJNOs = df_evts_plot_JJNOs.join(df_evts[['id', 'startDate']].set_index('id'), on='id')
fig5 = px.bar(df_evts_plot_JJNOs, x="startDate", y='country', color=fuc_of_ty,
              color_discrete_map=col_sel,
              labels={
                      "startDate": "Year of the Games",
                      "country": "Number of JJNOs"
                      })
st.plotly_chart(fig5)

df_medal = df_ms[['country', 'rank', 'name']].groupby(['country', 'rank']).count().reset_index()
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
df_map['country'] = df_ms['country_code'].value_counts().index
df_map['counts'] = df_ms['country_code'].value_counts().values
data = dict(type='choropleth',
            locations=df_map['country'], z=df_map['counts'])

layout = dict(title='Participating JJNOs',
              geo=dict(showframe=True,
                       projection={'type': 'robinson'}))
x = pg.Figure(data=[data], layout=layout)
st.plotly_chart(x)