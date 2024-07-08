'''shows an overview of all categories that are there (and were there)


'''
import streamlit as st
import plotly.express as px
import pandas as pd
import datetime as dt
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


st.title("Categories")
add_logo()

COLOR_MAP_CON = st.session_state['COLOR_MAP_CON']
COLOR_MAP = st.session_state['COLOR_MAP']

if 'df_total' in st.session_state:
    df_total = st.session_state['df_total']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")

if 'df_par' in st.session_state:
    df_par = st.session_state['df_par']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")


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
    st.write(df_par[['name', 'category_name', 'cat_type']].groupby(['category_name', 'cat_type']).count().reset_index())

df_cats_jjnos = df_par[['country', 'category_name', 'cat_type', 'continent']].groupby(['category_name', 'cat_type', 'continent']).nunique().reset_index()
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