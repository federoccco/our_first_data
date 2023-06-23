import pandas as pd
import numpy as np 
import streamlit as st
from streamlit_gsheets import GSheetsConnection

from geopy.geocoders import Nominatim

from streamlit_extras.let_it_rain import rain
from streamlit_extras.metric_cards import style_metric_cards

import plotly.express as px

import pydeck as pdk
import base64




# ----------------------------PAGE CONFIG ----------------------------
st.set_page_config(
    page_title="Our First Dat-a",
    page_icon=":heart:",
    layout="centered",
    initial_sidebar_state="collapsed",
)



# ---------------------------- SIDE BAR PAGES ----------------------------
page = st.sidebar.selectbox('Select page', ['Home', 'Tvseries & Anime', 'Movies', 'Games', 'Travel'])


# ---------------------------- CONTAINER CONFIG ---------------------------
series_header_container = st.container()
series_plot_container = st.container()

movies_header_container = st.container()
movies_plot_container = st.container()

games_header_container = st.container()
games_plot_container = st.container()

travel_header_container = st.container()
travel_plot_container = st.container()






# ---------------------------- DATA ----------------------------
conn = st.experimental_connection("gsheets", type=GSheetsConnection)



# --- SERIES ---
series_url = "https://docs.google.com/spreadsheets/d/1gHIV9R3v1jAg_FalmXAsm1FejaAZRVrUmlG8rhZM3WU/edit#gid=0"
data = conn.read(spreadsheet=series_url, usecols=[0, 1, 2, 3, 4])
series_df = pd.DataFrame(data)




# --- MOVIES --- 
movies_url = 'https://docs.google.com/spreadsheets/d/1gHIV9R3v1jAg_FalmXAsm1FejaAZRVrUmlG8rhZM3WU/edit#gid=1165920976'
data = conn.read(spreadsheet=movies_url, usecols=[0, 1, 2])
movies_df = pd.DataFrame(data)




# --- GAMES --- 
games_url = 'https://docs.google.com/spreadsheets/d/1gHIV9R3v1jAg_FalmXAsm1FejaAZRVrUmlG8rhZM3WU/edit#gid=1798755012'
data = conn.read(spreadsheet=games_url, usecols=[0, 1, 2, 3, 4])
games_df = pd.DataFrame(data)





# --- TRAVEL --- 
travel_url = 'https://docs.google.com/spreadsheets/d/1gHIV9R3v1jAg_FalmXAsm1FejaAZRVrUmlG8rhZM3WU/edit#gid=80335954'
data = conn.read(spreadsheet=travel_url, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
travel_df = pd.DataFrame(data)

# filling string columns
travel_columns_to_fill = ['data', 'mese', 'anno', 'spesa', 'regione']
travel_fill_values = {col: 'Uknown' if col in ['data', 'mese', 'anno', 'regione'] else 0 for col in travel_columns_to_fill}

travel_df.fillna(travel_fill_values, inplace=True) 



# ----------------------------------- HOME -----------------------------------------------
if page == 'Home':

    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"]{
    background-image: url("https://cdn.pixabay.com/photo/2018/12/17/08/10/sunset-3879934_1280.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    }

    [data-testid="stHeader"]{
    background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"]{
    right: 2rem;
    }

    [data-testid="stSidebar"]{
    background-color: rgba(0, 0, 0, 0);
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title('Our First Dat-a')
    st.write('Ti sei mai chiesta quante cose abbiamo fatto? No? Ecco le risposte alle domande che non avevi')






# -----------------------------------SERIES -----------------------------------------------

if page == 'Tvseries & Anime':
    rain(
    emoji="📷",
    font_size=54,
    falling_speed=5,
    animation_length=1,
)


    # ---header---
    with series_header_container:
        st.write('# Tvseries & Anime')
        '---'
        st.write('#### Qui puoi vedere le nostre statistiche riguardo SerieTv & Anime')
    
    # --- body ---
    with series_plot_container:
        
        #---filters lists---
        year_options = series_df['anno'].unique().tolist()
        completed_options = series_df['completato'].unique().tolist()

        # ---setting multiselection filter ---
        year = st.multiselect('Quali anni ti interessa vedere', year_options, default=year_options, key='year_options')
        complete = st.multiselect('E\' una serie completata oppure no?', completed_options, default=completed_options, key='completed_options')

        # new dataframe dynamically change with filters applied
        year_cond = series_df['anno'].isin(year)
        complete_cond = series_df['completato'].isin(complete)
        multi_series_df = series_df[year_cond & complete_cond]

        # --- metrics ---
        col1, col2, col3 = st.columns((3))
        col1.metric(label="Serie viste", value=multi_series_df['titolo'].count())
        col2.metric(label="Episodi totali", value=multi_series_df['episodi visti'].sum())
        col3.metric(label='Percentuale sul totale', value=f"{round(multi_series_df['titolo'].count() / series_df.shape[0]*100, 2)}%")

        style_metric_cards("""
            background_color: str = '#FFF'
            , border_size_px: int = 1
            , border_color: str = '#CCC'
            , border_radius_px: int = 5
            , border_left_color: str = '#9AD8E1'
            , box_shadow: bool = True"""
        )

        # --- bar chart ---

        df = multi_series_df['tipologia'].value_counts().reset_index()
        df.columns = ['Tipologia', 'Conteggio']

        fig = px.bar(df, x='Tipologia', y='Conteggio', text='Conteggio')
        fig.update_traces(textposition=['outside', 'outside'])

        st.plotly_chart(fig, theme='streamlit')

        '---'
        # ---line chart --- 
        df = multi_series_df.groupby(['anno', 'tipologia'])['episodi visti'].sum().reset_index()
        fig = px.line(df, x='anno', y='episodi visti', color='tipologia')

        # Traces
        fig.update_traces(line=dict(width=4))
        # Layout
        fig.update_layout(
            margin=dict(l=1,r=1,b=1,t=1)
            ,
            # Y axis
            yaxis=dict(
                title='Numero Episodi',
            ),
            # X axis            
            xaxis=dict(
                title=None,
                tickvals=year_options,
            ),
        )
        st.plotly_chart(fig, theme='streamlit')

    st.write(multi_series_df)






# -----------------------------------MOVIES -----------------------------------------------

if page == 'Movies':
    rain(
    emoji="🎬",
    font_size=54,
    falling_speed=5,
    animation_length=1,
)
    

    # ---header---
    with series_header_container:
        st.write('# Movies')
        '---'
        st.write('#### Qui puoi veder le nostre statistiche riguardo i Films')
    
    # --- body ---
    with series_plot_container:

        #---filters lists---
        year_options = movies_df['anno'].unique().tolist()
        genre_options = movies_df['genere'].unique().tolist()

        # ---setting multiselection filter ---
        year = st.multiselect('Quale anno ti interessa vedere?', year_options, default=year_options, key='year_options')
        genre = st.multiselect('Che genere?', genre_options, default=genre_options, key='genre_options')

        # new dataframe dynamically change with filters applied
        year_cond = movies_df['anno'].isin(year)
        genre_cond = movies_df['genere'].isin(genre)
        multi_movies_df = movies_df[year_cond & genre_cond]

        # --- metrics ---
        col1, col2 = st.columns(2)
        col1.metric(label="Film visti", value=multi_movies_df['titolo'].count())
        col2.metric(label="Generi visti", value=multi_movies_df['genere'].nunique())



        style_metric_cards("""
            background_color: str = '#FFF'
            , border_size_px: int = 1
            , border_color: str = '#CCC'
            , border_radius_px: int = 5
            , border_left_color: str = '#9AD8E1'
            , box_shadow: bool = True"""
        )

    # --- plots ---
    df = multi_movies_df['genere'].value_counts()
    fig = px.bar(df, x=df.index, y='count', text='count', title='Generi guardati')
    fig.update_traces(textposition=['outside' for _ in genre_options])
    st.plotly_chart(fig, theme='streamlit')


    st.write(multi_movies_df)






# ----------------------------------- GAMES -----------------------------------------------
if page == 'Games':
    rain(
    emoji="🎮",
    font_size=54,
    falling_speed=5,
    animation_length=1,
)
    # ---header ---
    with games_header_container:
        st.write('# Games')
        '---'
        st.write('#### Qui puoi vedere le nostre statistiche riguardo i Giochi')

    # --- body ---     
    with games_plot_container:

        #---filters lists---
        year_options = games_df['anno'].unique().tolist()
        genre_options = games_df['genere'].unique().tolist()
        finished_options = games_df['finito'].unique().tolist()

        # setting multiselection filter
        year = st.multiselect('Quali anni ti interessa vedere', year_options, default=year_options, key='games_year_options')
        genre = st.multiselect('Quale genere ti interessa vedere?', genre_options, default=genre_options, key='games_genre_options')
        finish = st.multiselect('Gioco completato?', finished_options, default=finished_options, key='games_finished_options')

        # conditions
        year_cond = games_df['anno'].isin(year)
        genre_cond = games_df['genere'].isin(genre)
        finish_cond = games_df['finito'].isin(finish)

        # newdf
        multi_games_df = games_df[year_cond & genre_cond & finish_cond]



        # --- metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric(label='Giochi giocati:', value=multi_games_df['titolo'].count())
        col2.metric(label='Generi giocati:', value=multi_games_df['genere'].nunique())
        col3.metric(label='Ore totali giocate:', value=multi_games_df['durata'].sum())

        style_metric_cards("""
            background_color: str = '#FFF'
            , border_size_px: int = 1
            , border_color: str = '#CCC'
            , border_radius_px: int = 5
            , border_left_color: str = '#9AD8E1'
            , box_shadow: bool = True"""
        )
    

    st.write(multi_games_df)


# ----------------------------------- TRAVEL -----------------------------------------------
if page == 'Travel':
    rain(
    emoji="🍕",
    font_size=54,
    falling_speed=5,
    animation_length=1,
)
    # ---header --- 
    with travel_header_container:
        st.write('# Travel')
        '---'
        st.write('#### Qui puoi vedere le nostre statistiche riguardo i Ristoranti')
    with travel_plot_container:

        #---filters lists---
        type_options = sorted(travel_df['tipo'].unique().tolist(), key=str.casefold)
        city_options = sorted(travel_df['citta'].unique().tolist(), key=str.casefold)
        region_options = sorted(travel_df['regione'].unique().tolist(), key=str.casefold)
        country_options = sorted(travel_df['stato'].unique().tolist(), key=str.casefold)
        
        travel_df['mese'] = travel_df['mese'].apply(lambda x: int(x) if x != 'Uknown' and pd.notnull(x) else x)
        month_options = travel_df['mese'].unique().tolist()
        month_options = sorted(month_options, key=lambda x: (x != 'Uknown', float('inf') if x == 'Uknown' else x) )


        travel_df['anno'] = travel_df['anno'].apply(lambda x: int(x) if x != 'Uknown' and pd.notnull(x) else x)
        travel_year_options = travel_df['anno'].unique().tolist()
        travel_year_options = sorted(travel_year_options, key=lambda x: (x != 'Uknown', float('inf') if x == 'Uknown' else x) )

        # setting multiselection filter
        travel_type = st.multiselect('Che tipologia ti interessa vedere?', type_options, default=type_options, key='travel_type_options')
        travel_city = st.multiselect('Quale città ti interessa vedere?', city_options, default=city_options, key='travel_city_options')
        travel_region = st.multiselect('Quale regione ti interessa vedere?', region_options, default=region_options, key='travel_region_options')
        travel_country = st.multiselect('Quale stato ti interessa vedere?', country_options, default=country_options, key='travel_country_options')
        travel_month = st.multiselect('Quali mesi ti interessa vedere?', month_options, default=month_options, key='travel_month_options')
        travel_year = st.multiselect('Quali anni ti interessa vedere?', travel_year_options, default=travel_year_options, key='travel_year_options')
        
        # new dataframe dynamically change with filters applied
        travel_type_cond = travel_df['tipo'].isin(travel_type)
        travel_city_cond = travel_df['citta'].isin(travel_city)
        travel_region_cond = travel_df['regione'].isin(travel_region)
        travel_country_cond = travel_df['stato'].isin(travel_country)
        travel_month_cond = travel_df['mese'].isin(travel_month)
        travel_year_cond = travel_df['anno'].isin(travel_year)



        # Dybamic dataframe
        multi_travel_df = travel_df[travel_type_cond & travel_city_cond & travel_region_cond & travel_country_cond & travel_month_cond & travel_year_cond]

        # Replace decimal commas with decimal points in the 'latitude' and 'longitude' columns
        multi_travel_df['latitude'] = multi_travel_df['latitude'].str.replace(',', '.')
        multi_travel_df['longitude'] = multi_travel_df['longitude'].str.replace(',', '.')
        multi_travel_df['spesa'] = multi_travel_df['spesa'].str.replace(',', '.')

        # casting float columns 
        multi_travel_df['latitude'] = multi_travel_df['latitude'].astype('float')
        multi_travel_df['longitude'] = multi_travel_df['longitude'].astype('float')
        multi_travel_df['spesa'] = multi_travel_df['spesa'].astype('float')




        # --- metrics ---

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label='Ristoranti unici visitati:', value=multi_travel_df['luogo'].nunique())
            st.metric(label='Visite totali:', value=multi_travel_df['luogo'].count())
        with col2:
            st.metric(label='Spesa totale:', value=f"{multi_travel_df['spesa'].sum()} €")
            st.metric(label='Tipologie provate:', value=multi_travel_df['tipo'].nunique())


        style_metric_cards("""
            background_color: str = '#FFF'
            , border_size_px: int = 1
            , border_color: str = '#CCC'
            , border_radius_px: int = 5
            , border_left_color: str = '#9AD8E1'
            , box_shadow: bool = True"""
        )


        # --- map ---

        # Group the DataFrame by latitude and longitude, count the occurrences of each pair and summing the amount spent
        grouped_df = multi_travel_df.groupby(['luogo', 'latitude', 'longitude']).agg(conteggio=('luogo','count'), spesa_totale=('spesa','sum')).reset_index()

        # view
        view = pdk.ViewState(latitude=45.44693, longitude=8.6221612, zoom=12, pitch=50)

        # column layer
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=grouped_df,
            get_position=["longitude", "latitude", "conteggio"],
            get_elevation="conteggio",
            get_radius=10000,
            elevation_scale=500,
            elevation_range=[0, 2000],
            radius=100,
            get_fill_color=["latitude * 10", "latitude", "latitude * 10", 140],
            pickable=True,
            auto_highlight=True,
        )

        # tooltip
        tooltip = {
            "html": "<b>{luogo}</b> : <b>{conteggio}</b> volta/e per un totale di {spesa_totale} Euro",
            "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        }

        # chart
        st.pydeck_chart(pdk.Deck(
            column_layer,
            initial_view_state=view,
            tooltip=tooltip,
            map_provider="mapbox",
            map_style=None,
        ))
        

        '---'
        st.write('**Ristoranti visitati al mese**')

        # --- line chart --- 

        filtered_df = multi_travel_df[(multi_travel_df['anno'] != 'Uknown') & (multi_travel_df['mese'] != 'Uknown')]

        # setting concat time column
        filtered_df['time'] = pd.to_datetime(filtered_df['anno'].astype(str) + '-' + filtered_df['mese'].astype(str))

        # Group by 'time' and count the occurrences of 'luogo'
        line_grouped_df = filtered_df.groupby('time')['luogo'].count().reset_index()
        line_grouped_df['time'] = line_grouped_df['time'].dt.strftime('%m-%Y')
        
        # Sort the DataFrame by the 'time' column in the desired order
        line_grouped_df['time'] = pd.to_datetime(line_grouped_df['time'], format='%m-%Y')  # Convert 'time' column back to datetime format
        line_grouped_df.sort_values('time', inplace=True)
        line_grouped_df['time'] = line_grouped_df['time'].dt.strftime('%m-%Y')  # Convert 'time' column back to string format


        # Set the 'time' column as the index for sorting
        line_grouped_df.set_index('time', inplace=True)

        # chart  
        fig = px.line(line_grouped_df, x=line_grouped_df.index, y='luogo')

        # Traces
        fig.update_traces(line=dict(width=4))
        # Layout
        fig.update_layout(
            margin=dict(l=1,r=1,b=1,t=1)
            ,
            # Y axis
            yaxis=dict(
                title='Conteggio',
            ),
            # X axis            
            xaxis=dict(
                title=None,
                tickvals=line_grouped_df.index,
            ),
        )
        st.plotly_chart(fig, theme='streamlit')


        '---'

        # --- bar plot ---
        bar_travel_df = multi_travel_df['tipo'].value_counts()
        fig = px.bar(bar_travel_df, x=bar_travel_df.index, y='count', text='count', title='Conteggio delle tipologie di ristoranti')
        fig.update_traces(textposition=['outside' for _ in type_options])

        st.plotly_chart(fig, theme='streamlit')


        # --- print the table ---
        st.write(multi_travel_df[['luogo', 'tipo', 'spesa']])
        