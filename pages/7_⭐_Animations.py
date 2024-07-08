'''
 some animation of the nations with the most participants
'''
import streamlit as st
import plotly.express as px


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


st.title("Animations")
add_logo()

if 'df_time_smooth' in st.session_state:
    df_time_smooth = st.session_state['df_time_smooth']
else:
    st.error("Go back to 🏠 Main Page and wait until data is loaded")


COLOR_MAP_CON = st.session_state['COLOR_MAP_CON']
COLOR_MAP = st.session_state['COLOR_MAP']

# only adults
df_time_smooth1 = df_time_smooth[df_time_smooth['age_division'] == "Adults"]
df_animated_bar = df_time_smooth1[['month', 'country', 'name', 'continent']].groupby(['month', 'country', 'continent']).count().reset_index()

# some bug in plotly does not show all colors if they are no in the first frame.
# https://github.com/plotly/plotly.py/issues/2259
# workaround: Add "fake data for first frame:
fake_africa = {'month': df_animated_bar['month'].min(), 'country': ' ', 'continent': 'Africa', 'name': 0}
df_animated_bar.loc[len(df_animated_bar)] = fake_africa

fake_asia = {'month': df_animated_bar['month'].min(), 'country': ' ', 'continent': 'Asia', 'name': 0}
df_animated_bar.loc[len(df_animated_bar)] = fake_asia

fake_panam = {'month': df_animated_bar['month'].min(), 'country': ' ', 'continent': 'Pan America', 'name': 0}
df_animated_bar.loc[len(df_animated_bar)] = fake_panam

fake_oceania = {'month': df_animated_bar['month'].min(), 'country': ' ', 'continent': 'Oceania', 'name': 0}
df_animated_bar.loc[len(df_animated_bar)] = fake_oceania

figan = px.bar(df_animated_bar,
               y='country',
               x='name',
               color='continent',
               color_discrete_map=COLOR_MAP_CON,
               animation_frame=df_animated_bar.month.astype(str),
               orientation='h',
               text='country',
               title="Largest number of participants (only Adults, all disciplines)",
               labels={
                        "month": "Date [year]",
                        "name": "Number of Athletes"
                        })

figan.layout.updatemenus[0].buttons[0]['args'][1]['frame']['duration'] = 300

figan.update_yaxes(visible=False)
figan.update_yaxes(categoryorder='total descending')
figan.update_yaxes(range=(-.5, 9.5))
st.plotly_chart(figan)

# largest categories
df_animated_bar_cat = df_time_smooth[['month', 'category_name', 'name', 'cat_type']].groupby(['month', 'category_name', 'cat_type']).count().reset_index()

# some bug in plotly does not show all colors if they are no in the first frame.
# https://github.com/plotly/plotly.py/issues/2259
# workaround: Add "fake data for first frame:
fake_jiu = {'month': df_animated_bar_cat['month'].min(), 'category_name': ' ', 'cat_type': 'Jiu-Jitsu', 'name': 0}
df_animated_bar_cat.loc[len(df_animated_bar_cat)] = fake_jiu

fake_show = {'month': df_animated_bar_cat['month'].min(), 'category_name': ' ', 'cat_type': 'Show', 'name': 0}
df_animated_bar_cat.loc[len(df_animated_bar_cat)] = fake_show

fake_duo = {'month': df_animated_bar_cat['month'].min(), 'category_name': ' ', 'cat_type': 'Duo', 'name': 0}
df_animated_bar_cat.loc[len(df_animated_bar_cat)] = fake_duo

figan1 = px.bar(df_animated_bar_cat,
                y='category_name',
                x='name',
                color='cat_type',
                color_discrete_map=COLOR_MAP,
                animation_frame=df_animated_bar_cat.month.astype(str),
                orientation='h',
                text='category_name',
                title="Largest number of participants in category",
                labels={
                        "month": "Date [year]",
                        "name": "Number of Athletes"
                        })

figan1.layout.updatemenus[0].buttons[0]['args'][1]['frame']['duration'] = 300

figan1.update_yaxes(visible=False)
figan1.update_yaxes(categoryorder='total descending')
figan1.update_yaxes(range=(-.5, 9.5))
st.plotly_chart(figan1)