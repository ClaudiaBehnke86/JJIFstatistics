''' This mode show all events that JJIF ever made

One can get an idea which countries organized events and how the number of participants was
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


# Retrieve the data from session state
if 'df_evts' in st.session_state:
    df_evts = st.session_state['df_evts']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")
COLOR_MAP_ETYPE = st.session_state['COLOR_MAP_ETYPE']
dend = st.session_state['dend']

st.title("Events Overview")
add_logo()
st.write("This page show all events that JJIF ever made")
# show events on screen

with st.expander("Details of events"):
    st.write(df_evts)

# shows a bar chat with number of participates of each event and on x axis the date of the event
df_dip_evts = df_evts[['Name Event', 'startDate', 'Number of Participants', 'eventtype', 'country']]
fig1 = px.bar(df_dip_evts, x='startDate', y='Number of Participants', color='eventtype',
              hover_data=['startDate', 'Name Event', 'Number of Participants', 'country'],
              title="Events Organised",
              color_discrete_map=COLOR_MAP_ETYPE,
              labels={
                        "dates": "Date [year]",
                        "name": "Name of event"
                        }
              )
fig1.update_layout(xaxis_range=[df_dip_evts['startDate'].min(), dend])
# make the bars wider and visible
fig1.update_traces(width=2404800000)
st.plotly_chart(fig1)

st.success(" :arrow_left: Go to Single Event to see details per event")

# show a map of number of events organized per country
df_map1 = pd.DataFrame()
df_map1['country'] = df_evts['country_code'].value_counts().index
df_map1['counts'] = df_evts['country_code'].value_counts().values

data = dict(type='choropleth',
            locations=df_map1['country'], z=df_map1['counts'])

layout = dict(title='Countries that organized Events',
              geo=dict(showframe=True,
                       projection={'type': 'robinson'}))
x = pg.Figure(data=[data], layout=layout)
x.update_geos(
        showcountries=True, countrycolor="black"
)
st.plotly_chart(x)
