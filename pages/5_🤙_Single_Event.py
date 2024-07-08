'''
Details of individual events

'''
import plotly.graph_objs as pg
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


st.title("Details for one event")
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

COLOR_MAP_CON = st.session_state['COLOR_MAP_CON']
COLOR_MAP = st.session_state['COLOR_MAP']

evt_sel = df_evts['Name Event'].unique()
evtt_select = st.selectbox("Select the event:",
                           evt_sel, help="Type the name and year of the event you are looking for")
select_id = df_evts['id'][df_evts['Name Event'] == evtt_select].iloc[0]


df_single_event = df_par[df_par['id'] == select_id]

df_cats_jjnos = df_single_event[['name', 'category_name', 'cat_type', 'continent']].groupby(['category_name', 'cat_type', 'continent']).nunique().reset_index()
fig_cats_jjnos = px.bar(df_cats_jjnos, x="category_name", y="name",
                        color="continent", title="Athletes per category",
                        color_discrete_map=COLOR_MAP_CON,
                        labels={
                                "category_name": "Category Name",
                                "name": "Number of Athletes"})
fig_cats_jjnos.update_layout(xaxis={'categoryorder': 'category ascending'})

st.write("In total ", len(df_single_event['name'].unique()), "Athletes from",
         len(df_single_event['country'].unique()), "JJNOs")
st.plotly_chart(fig_cats_jjnos)

df_numb_cat = df_single_event[['name', 'country', 'category_name']].groupby(['category_name']).nunique().reset_index()

with st.expander("Show numbers"):
    df_numb_cat

left_column, right_column = st.columns(2)
with left_column:
    df_cat = pd.DataFrame()
    df_cat['cat_type'] = df_single_event['cat_type'].value_counts().index
    df_cat['counts'] = df_single_event['cat_type'].value_counts().values
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
    df_gender['gender'] = df_single_event['gender'].value_counts().index
    df_gender['counts'] = df_single_event['gender'].value_counts().values
    fig2 = px.pie(df_gender, values='counts', names='gender',
                  color='gender',
                  color_discrete_map={"Women": 'rgb(243, 28, 43)',
                                      "Men": 'rgb(0,144,206)',
                                      "Mixed": 'rgb(211,211,211)',
                                      "Open": 'rgb(105,105,105)'
                                      },
                  title='Gender distribution')
    st.plotly_chart(fig2, use_container_width=True)

df_medal = df_single_event[['country', 'rank', 'name']].groupby(['country', 'rank']).count().reset_index()
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

df_map = pd.DataFrame()
df_map['country'] = df_single_event['country_code'].value_counts().index
df_map['counts'] = df_single_event['country_code'].value_counts().values
data = dict(type='choropleth',
            locations=df_map['country'], z=df_map['counts'])

layout = dict(title='Participating JJNOs',
              geo=dict(showframe=True,
                       projection={'type': 'robinson'}))
x = pg.Figure(data=[data], layout=layout)
st.plotly_chart(x)
