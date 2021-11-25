import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show
from bokeh.tile_providers import get_provider, CARTODBPOSITRON_RETINA, STAMEN_TONER, STAMEN_TERRAIN
from bokeh.models import HoverTool, FreehandDrawTool, BoxEditTool, ColumnDataSource, ColorBar
from bokeh.palettes import Plasma10, Spectral11
from bokeh.transform import linear_cmap

#=======================================================================================================
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
        color: #FFFFFF;
        background-color: #042A37;
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

#=======================================================================================================
# Define Functions and Global Variables
filepath = os.getcwd()

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

@st.cache
def load_data(n):
    if n == 'map_data' :
        data = pd.read_csv(filepath+'/assets/APP_data_all.csv', index_col='PID')
    elif n == 'house_data' :
        data = pd.read_csv(filepath+'/assets/cleaned_data.csv', index_col='PID')
    return data
all_data = load_data('map_data')

def plot_stacked(s_data, overlay=None, all_data=all_data):
    sec_order=['NW','SO','WE','SE','NO','DT']
    fig, ax1 = plt.subplots()
    s_data.loc[:,sec_order].T.plot(ax=ax1, kind='bar', rot=0, width=0.8,
                            stacked=True, figsize=(10,6)).legend(bbox_to_anchor=(1.051, 1.0))
    ax1.set_ylabel('Proportion')
    ax2 = ax1.twinx()
    sns.stripplot(ax=ax2, x='Sector', y=overlay, data=all_data, order=sec_order, color='0.6', edgecolor='k', linewidth=0.5)
    return fig

#=======================================================================================================
# Navigation
st.sidebar.image(filepath+'/assets/App_Logo.jpg', use_column_width=True) 
page = st.sidebar.radio("Navigation", ["Map of Ames", "City Sectors", "P3", "P4", "P5", "Collaborators"]) 

#------------------------------------------------------------------------------------------------------
# Page1: Map of Ames, IA
if page == "Map of Ames":
    with st.container():
        st.title('Map of Ames')
        col1, col2 = st.columns([3, 1]) #Set Columns

        # Misc Map Settings
        background = get_provider(STAMEN_TERRAIN)

        # Sidebar Radio Button
        # For selecting map plot
        map_choice = col2.radio("Choose Map:", ('SalePrice', 'Neighborhood', 'Sector'))

        # Mapper Function
        def bok_fig():
            # Base Map Layer
            fig = figure(plot_width=940, plot_height=700,
                        x_range=(Ames_center[0]-8000, Ames_center[0]+3000), 
                        y_range=(Ames_center[1]-8000, Ames_center[1]+5000),
                        x_axis_type="mercator", y_axis_type="mercator",
                        title="Ames Iowa Housing Map")
            fig.add_tile(background)
            return fig

        def bok_layer(map_choice = map_choice, fig = bok_fig()):
            # Set map data, hover tool, and color palette
            if map_choice == 'SalePrice':
                mycolors = linear_cmap(field_name='SalePrice', palette=Plasma10[::-1], low=min(all_data.SalePrice) ,high=max(all_data.SalePrice))
                color_bar = ColorBar(color_mapper=mycolors['transform'], width=8,  location=(0,0),title="Price $(thousands)")
                fig.add_layout(color_bar, 'right')
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('Price', '@SalePrice')]
                fig.add_tools(my_hover)
            elif map_choice == 'Neighborhood':
                mycolors = linear_cmap(field_name='le_Neighbor', palette=Spectral11[::-1], low=min(all_data.le_Neighbor) ,high=max(all_data.le_Neighbor))
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('', '@Neighborhood')]
                fig.add_tools(my_hover)
            else:
                mycolors = linear_cmap(field_name='le_Sector', palette=Spectral11[::-1], low=min(all_data.le_Sector) ,high=max(all_data.le_Sector))    
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('', '@Neighborhood')]
                fig.add_tools(my_hover)

            # Dots for Houses
            fig.circle(x="x_merc", y="y_merc",
                    size=7,
                    fill_color=mycolors, line_color='black',
                    fill_alpha=0.7,
                    name='House',
                    source=all_data)
            

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

        col1.write(f'Data: {map_choice}')
        col1.bokeh_chart(bok_layer())

        with col1.expander("Sidenote on Distance from Walmart vs YearBuilt"):
            st.write("""
                Distance from Walmart correlates with YearBuilt? (R2 = 0.7)
            """)
            st.image(filepath+'/assets/Walmart_YrBuilt.png')

        with col1.expander("Ames Visitor Map"):
            st.write("""
                City Sectors from The Ames Convention & Visitors Bureau
            """)
            st.image(filepath+'/assets/Ames.png')

#------------------------------------------------------------------------------------------------------
# Page 2 City Sector EDA
elif page == "City Sectors":
    with st.container():
        st.title('EDA with City Sectors')
        col1, col2 = st.columns([3, 1]) #Set Columns
        sns.set_palette('gist_earth')

        # percentage of houseClass in each Sector of city
        stack_data = all_data.groupby(['Sector'])['MSSubClass'].value_counts(normalize=True).to_frame()
        stack_data.rename(columns={'MSSubClass':'HouseType'}, inplace=True)
        stack_data.reset_index(inplace=True)
        stack_data = stack_data.pivot(index='MSSubClass',columns='Sector', values='HouseType')

        overlay_choice = col2.radio("Overlay Data:", ('SalePrice', 'YearBuilt', 'OverallQual'))

        col1.pyplot(plot_stacked(stack_data, overlay_choice))

        with col1.expander("HouseType Comparisons"):
            st.write("""
                
            """)
            st.image(filepath+'/assets/HouseType.png')

        with col1.expander("Price per SF Analysis"):
            st.image(filepath+'/assets/PperSF.png')
            st.write("Price per SF drops as house size increases in all Sectors, but most pronounced in SE, NO, & DT.")
            st.write("The phenomenon is only seen in Split, Duplex or 2 Family houses.")

elif page == "P3":
    # Display details of page 2
    st.title('Feature selection')

    data_load_state = st.text('Loading data...')
    graph_data = load_data('house_data')
    selected = st.selectbox(
         'Choose a feature:',
         ('Fireplaces', 'FireplaceQu', 'GarageCars', 'CentralAir', 'HeatingQC', 'OverallQual',))
    fig = px.scatter(graph_data,x='GrLivArea',y='SalePrice',facet_col=selected,color=selected,trendline='ols')
    st.plotly_chart(fig)
  
elif page == "P4":
    # Display details of page 2
    st.title('Page 4')

elif page == "P5":
    # Display details of page 2
    st.title('Page 5')

elif page == "Collaborators":
    # Display details of page 2
    st.title('Collaborators')
    st.subheader('Daniel Nie')
    st.subheader('David Kressley')
    st.subheader('Karl Lundquist')
    st.subheader('Tony Pennoyer')