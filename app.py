import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
# Define Global Functions and Variables
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
def load_data(what_data):
    if what_data == 'map_data' :
        data = pd.read_csv(filepath+'/assets/APP_data_all.csv', index_col='PID')
    elif what_data == 'house_data' :
        data = pd.read_csv(filepath+'/assets/cleaned_data.csv', index_col='PID')
    elif what_data == 'pickle_data' :
        data = pd.read_csv(filepath+'/assets/pickle_base.csv')
    return data
map_data = load_data('map_data')
pkl_data = load_data('pickle_data')

def plot_stacked(s_data, overlay=None, m_data=map_data):
    sec_order=['NW','SO','WE','SE','NO','DT']
    fig, ax1 = plt.subplots()
    s_data.loc[:,sec_order].T.plot(ax=ax1, kind='bar', rot=0, width=0.8,
                            stacked=True, figsize=(10,6)).legend(bbox_to_anchor=(1.051, 1.0))
    ax1.set_ylabel('Proportion')
    ax2 = ax1.twinx()
    sns.stripplot(ax=ax2, x='Sector', y=overlay, data=m_data, order=sec_order, color='0.6', edgecolor='k', linewidth=0.5)
    return fig

# ========Modeling Functions===================================================
pkl_model = pickle.load(open(filepath+'/assets/APP_model.pkl', 'rb'))

def num_format(num):
    # converts any int/float to human readable string with thousandth commas
    new_num = ''
    for idx, c in enumerate(str(np.int64(num))[::-1]):
        if (idx+1)%4 == 0:
            new_num += ','
        new_num += c
    return new_num[::-1]

def pkl_dum_encode_nbr(base_data, code):
    # Encodes the neighborhood selected with 1, all other dummy columns are set to 0
    target = 'Neighborhood_'+code
    neigh_cols = list(base_data.filter(regex='^Neigh').columns)
    base_data.loc[0,neigh_cols] = 0
    if target in neigh_cols:
        neigh_cols.remove(target)
        base_data.loc[0,target] = 1
    return base_data

def pkl_dum_encode_type(base_data, code):
    # Encodes the HouseType selected with 1, all other dummy columns are set to 0
    target = 'MSSubClass_'+code
    type_cols = list(base_data.filter(regex='^MSSub').columns)
    base_data.loc[0,type_cols] = 0
    if target in type_cols:
        type_cols.remove(target)
        base_data.loc[0,target] = 1
    return base_data
# =============================================================================

#=======================================================================================================
# Navigation
st.sidebar.image(filepath+'/assets/App_Logo.jpg', use_column_width=True) 
page = st.sidebar.radio("Navigation", ["Map of Ames", "City Sectors", "House Features", "P4", "Renovation Model", "Collaborators"]) 

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
                mycolors = linear_cmap(field_name='SalePrice', palette=Plasma10[::-1], low=min(map_data.SalePrice) ,high=max(map_data.SalePrice))
                color_bar = ColorBar(color_mapper=mycolors['transform'], width=8,  location=(0,0),title="Price $(thousands)")
                fig.add_layout(color_bar, 'right')
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('Price', '@SalePrice')]
                fig.add_tools(my_hover)
            elif map_choice == 'Neighborhood':
                mycolors = linear_cmap(field_name='le_Neighbor', palette=Spectral11[::-1], low=min(map_data.le_Neighbor) ,high=max(map_data.le_Neighbor))
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('', '@Neighborhood')]
                fig.add_tools(my_hover)
            else:
                mycolors = linear_cmap(field_name='le_Sector', palette=Spectral11[::-1], low=min(map_data.le_Sector) ,high=max(map_data.le_Sector))    
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('', '@Neighborhood')]
                fig.add_tools(my_hover)

            # Dots for Houses
            fig.circle(x="x_merc", y="y_merc",
                    size=7,
                    fill_color=mycolors, line_color='black',
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
        stack_data = map_data.groupby(['Sector'])['MSSubClass'].value_counts(normalize=True).to_frame()
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

#------------------------------------------------------------------------------------------------------
# Page 3 Feature Plots
elif page == "House Features":
    st.title('Feature selection')

    data_load_state = st.text('Loading data...')
    graph_data = load_data('house_data')
    selected = st.selectbox(
         'Choose a feature:',
         ('Fireplaces', 'FireplaceQu', 'GarageCars', 'CentralAir', 'HeatingQC', 'OverallQual',))
    fig = px.scatter(graph_data,x='GrLivArea',y='SalePrice',facet_col=selected,color=selected,trendline='ols',width=900, height=500,
        title = 'Sale Price vs. GrLivArea by ' + selected)
    st.plotly_chart(fig)
  
elif page == "P4":
    # Display details of page 4
    st.title('Page 4')

#------------------------------------------------------------------------------------------------------
# Page 4 Modeling
elif page == "Renovation Model":
    with st.container():
        st.title('Renovation Model Test')
        col_main, col_empty, col_b, col_bpx, col_r, col_rpx = st.columns([2,0.5,2,2,2,2]) #Set Columns
        col_main.markdown('#### ***Select:***')
        col_main.markdown('##### Location & Type of House')
        col_b.markdown('#### ** * **')
        col_b.markdown('##### Baseline House')
        col_r.markdown('#### ** * **')
        col_r.markdown('##### Renovation')
    
    with st.container():
        col_main, col_empty, col_b, col_bpx, col_r, col_rpx = st.columns([2,0.5,2,2,2,2]) #Set Columns

        #------Set Base Prediction House---------
        # Location & Type of House
        pkl_basehouse = pkl_data.copy()

        sec_select = col_main.selectbox('Select Sector',['Downtown','South','West','South East','North','North West'])
        sec_mapper = {'Downtown':'DT','South':'SO','West':'WE','South East':'SE','North':'NO','North West':'NW'}
        model_sec = sec_mapper[sec_select]
        model_neib = col_main.radio('Select Neighborhood',map_data.loc[map_data.Sector==model_sec]['Neighborhood'].unique())
        pkl_basehouse = pkl_dum_encode_nbr(pkl_basehouse, model_neib)

        model_htype = col_main.radio('Select Type of House',map_data.loc[map_data.Neighborhood==model_neib]['MSSubClass'].unique())
        pkl_basehouse = pkl_dum_encode_type(pkl_basehouse, model_htype)

        # Base House Details
        base_pool = col_b.radio('Pool',['No', 'Yes'])
        pkl_basehouse['HasPool'] = 0 if base_pool == 'No' else 1

        # Base House MODEL PRICE
        
        pkl_baseprice = np.floor(10**pkl_model.predict(pkl_basehouse)[0])
        col_bpx.subheader(f'**${num_format(pkl_baseprice)}**')
        col_bpx.caption('Baseline House Price')

        # RENO House Details
        pkl_renohouse = pkl_basehouse.copy()
        reno_pool = col_r.radio('Build Pool',['No', 'Yes'])

        pkl_renohouse['HasPool'] = 0 if reno_pool == 'No' else 1

        # RENOV House Details
        pkl_renoprice = np.floor(10**pkl_model.predict(pkl_renohouse)[0])
        col_rpx.subheader(f'**${num_format(pkl_renoprice)}**')
        col_rpx.caption('Renovated House Price')

        col_rpx.markdown(f'### **${num_format(pkl_renoprice - pkl_baseprice)}**')
        col_rpx.caption('Difference')

#------------------------------------------------------------------------------------------------------
# Page 5 About Page
elif page == "Collaborators":
    st.title('Collaborators')
    st.subheader('Daniel Nie')
    st.subheader('David Kressley')
    st.subheader('Karl Lundquist')
    st.subheader('Tony Pennoyer')