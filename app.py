import os
import streamlit as st
import pandas as pd
import numpy as np

from bokeh.plotting import figure, show
from bokeh.tile_providers import get_provider, CARTODBPOSITRON_RETINA, STAMEN_TONER, STAMEN_TERRAIN_RETINA
from bokeh.models import HoverTool, FreehandDrawTool, BoxEditTool, ColumnDataSource, ColorBar
from bokeh.palettes import Plasma10
from bokeh.transform import linear_cmap

#------------------------------------------------------------------------------------------------
# App CSS theme-ing
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1800px;
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: #000000;
        background-color: #7A8891;
    }}
    .reportview-container .css-ng1t4o {{
        color: #FFFFFF;
        background-color: #042A37;
        width: 250px;
        padding-top: 3rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

#------------------------------------------------------------------------------------------------
# Define Functions
def to_mercator(lat, lon):
    
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
        lat * (np.pi/180.0)/2.0)) * scale
    return (x, y)

landmarks = {'landmarks':['Iowa State University',
                          'Municipal Airport',
                          'North Grand Mall',
                          'Mary Greeley Medical Center',
                          'Jack Trice Stadium',
                          'Walmart Supercenter'],
            'x_merc':[to_mercator(42.0267,-93.6465)[0],
                      to_mercator(41.9987,-93.6223)[0],
                      to_mercator(42.0494,-93.6224)[0],
                      to_mercator(42.0323,-93.6111)[0],
                      to_mercator(42.0140,-93.6359)[0],
                      to_mercator(42.0160016, -93.6068719)[0]],
            'y_merc':[to_mercator(42.0267,-93.6465)[1],
                      to_mercator(41.9987,-93.6223)[1],
                      to_mercator(42.0494,-93.6224)[1],
                      to_mercator(42.0323,-93.6111)[1],
                      to_mercator(42.0140,-93.6359)[1],
                      to_mercator(42.0160016, -93.6068719)[1]]}
marks = pd.DataFrame(landmarks)

Ames_center = to_mercator(42.034534, -93.620369)

filepath = os.getcwd()
#--------------------------------------------------------------------------------------
# Navigation
st.sidebar.image(filepath+'/assets/App_Logo.jpg', use_column_width=True) 
page = st.sidebar.radio("Navigation", ["Map of Ames", "P2", "P3", "Authors"]) 

# APP Page1: Map of Ames, IA
if page == "Map of Ames":
    @st.cache
    def load_data():
        data = pd.read_csv(filepath+'/assets/APP_data.csv', index_col='PID')
        return data

    data_load_state = st.text('Loading data...')
    map_data = load_data()

    # Misc Map Settings
    background = get_provider(STAMEN_TERRAIN_RETINA)
    x_zoom = 7000
    y_zoom = 5000

    st.title('Initial EDA with Location Data')

    # Sidebar Radio Button
    # For selecting map plot
    map_choice = st.sidebar.radio("Choose Map:", ('SalePrice', 'Neighborhood', 'Sector'))

    # Mapper Function
    def bok_fig():
        # Base Map Layer
        fig = figure(plot_width=1000, plot_height=800,
                    x_range=(Ames_center[0]-x_zoom, Ames_center[0]+y_zoom), 
                    y_range=(Ames_center[1]-x_zoom, Ames_center[1]+y_zoom),
                    x_axis_type="mercator", y_axis_type="mercator",
                    title="Ames Iowa Housing Map")
        fig.add_tile(background)
        return fig

    def bok_layer(map_choice = map_choice, fig = bok_fig()):
        # Set map data, hover tool, and color palette
        if map_choice == 'SalePrice':
            mycolors = linear_cmap(field_name='SalePrice', palette=Plasma10[::-1], low=min(map_data.SalePrice) ,high=max(map_data.SalePrice))
            color_bar = ColorBar(color_mapper=mycolors['transform'], width=8,  location=(0,0),title="Price $(thousands)")
            fig.add_layout(color_bar, 'right')
            my_hover = HoverTool(names=['House'])
            my_hover.tooltips = [('Price', '@SalePrice')]
            fig.add_tools(my_hover)
        elif map_choice == 'Neighborhood':
            mycolors = linear_cmap(field_name='le_Neighbor', palette=Plasma10[::-1], low=min(map_data.le_Neighbor) ,high=max(map_data.le_Neighbor))
            my_hover = HoverTool(names=['House'])
            my_hover.tooltips = [('', '@Neighborhood')]
            fig.add_tools(my_hover)
        else:
            mycolors = linear_cmap(field_name='le_Sector', palette=Plasma10[::-1], low=min(map_data.le_Sector) ,high=max(map_data.le_Sector))    

        # Dots for Houses
        fig.circle(x="x_merc", y="y_merc",
                size=7,
                fill_color=mycolors, line_color=mycolors,
                fill_alpha=0.7,
                name='House',
                source=map_data)
        

        # Big Dots for Landmarks, with Hover interactivity
        my_hover = HoverTool(names=['landmark'])
        my_hover.tooltips = [('', '@landmarks')]
        fig.circle(x="x_merc", y="y_merc",
                size=18,
                fill_color="dodgerblue", line_color='dodgerblue',
                fill_alpha=0.4,
                name='landmark',
                source=marks)
        fig.add_tools(my_hover)

        return fig

    st.subheader(f'Map data: {map_choice}')
    st.bokeh_chart(bok_layer())

    data_load_state.text("Data loading Done! (st.cache)")

elif page == "P2":
    # Display details of page 2
    st.title('Page 2')

elif page == "P3":
    # Display details of page 2
    st.title('Page 3')

elif page == "Authors":
    # Display details of page 2
    st.title('About the Authors')
    st.subheader('Daniel Nie')
    st.subheader('David Kressley')
    st.subheader('Karl Lundquist')
    st.subheader('Tony Pennoyer')