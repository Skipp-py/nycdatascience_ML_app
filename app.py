import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, CARTODBPOSITRON_RETINA
from bokeh.models import HoverTool, ColumnDataSource, ColorBar, CustomJS, DataTable, TableColumn, HTMLTemplateFormatter
from bokeh.palettes import Plasma10, Spectral11
from bokeh.transform import linear_cmap
from streamlit_bokeh_events import streamlit_bokeh_events

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
    .reportview-container .css-ng1t4o {{
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

neib_fullname = {'Blmngtn':'Bloomington Heights',
       'Blueste':'Bluestem',
       'BrDale':'Briardale',
       'BrkSide':'Brookside',
       'ClearCr':'Clear Creek',
       'CollgCr':'College Creek',
       'Crawfor':'Crawford',
       'Edwards':'Edwards',
       'Gilbert':'Gilbert',
       'IDOTRR':'Iowa DOT and Rail Road',
       'MeadowV':'Meadow Village',
       'Mitchel':'Mitchell',
       'NAmes':'North Ames',
       'NoRidge':'Northridge',
       'NPkVill':'Northpark Villa',
       'NridgHt':'Northridge Heights',
       'NWAmes':'Northwest Ames',
       'OldTown':'Old Town',
       'SWISU':'South West of ISU',
       'Sawyer':'Sawyer',
       'SawyerW':'Sawyer West',
       'Somerst':'Somerset',
       'StoneBr':'Stone Brook',
       'Timber':'Timberland',
       'Veenker':'Veenker',
       'Greens':'Greensboro',
       'GrnHill':'Greens Hills',
       'Landmrk':'Landmark Villas'}

# Misc Map Settings
background = get_provider(CARTODBPOSITRON_RETINA)
# Mapper Function
def bok_fig(w=940, h=700, box_select=False):
    # Base Map Layer
    if not box_select:
        fig = figure(plot_width=w, plot_height=h,
                    x_range=(Ames_center[0]-8000, Ames_center[0]+3000), 
                    y_range=(Ames_center[1]-8000, Ames_center[1]+5000),
                    x_axis_type="mercator", y_axis_type="mercator",
                    title="Ames Iowa Housing Map")
    else:
        fig = figure(plot_width=w, plot_height=h,
                    x_range=(Ames_center[0]-8000, Ames_center[0]+3000), 
                    y_range=(Ames_center[1]-8000, Ames_center[1]+5000),
                    x_axis_type="mercator", y_axis_type="mercator",
                    title="Ames Iowa Housing Map",
                    tools="box_select", active_drag="box_select")
    fig.add_tile(background)
    return fig

@st.cache
def load_data(what_data):
    if what_data == 'map_data' :
        data = pd.read_csv(filepath+'/assets/APP_data_all.csv', index_col='PID')
    elif what_data == 'house_data' :
        data = pd.read_csv(filepath+'/assets/model_data.csv', index_col='PID')
    elif what_data == 'page_3_data' :
        data = pd.read_csv(filepath+'/assets/page_3_data.csv', index_col='PID')
    elif what_data == 'pickle_data' :
        data = pd.read_csv(filepath+'/assets/pickle_base.csv', index_col='PID')
    return data

map_data = load_data('map_data')
house_data = load_data('house_data')
pkl_data = load_data('pickle_data')
page_3_data = load_data('page_3_data')

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
pkl_model = pickle.load(open(filepath+'/assets/APP_model_CBR.pkl', 'rb'))

def num_format(num):
    # converts any int/float to human readable string with thousandth commas
    new_num = ''
    for idx, c in enumerate(str(np.int64(num))[::-1]):
        if (idx+1)%4 == 0:
            new_num += ','
        new_num += c
    return new_num[::-1]

def pkl_dum_encode(base_data, code, feat):
    # Encodes the feature selected with '1', all other dummy columns are set to '0'
    reg_text = '^'+feat
    target = feat+code
    feat_cols = list(base_data.filter(regex=reg_text).columns)
    base_data.loc[0,feat_cols] = 0
    if target in feat_cols:
        feat_cols.remove(target)
        base_data.loc[0,target] = 1
    return base_data

basehouse_PIN = 535454150
basehouse_medians = house_data.groupby(['Neighborhood','MSSubClass']).agg('median')

# =============================================================================

#=======================================================================================================
# Navigation
st.sidebar.image(filepath+'/assets/App_Logo.jpg', use_column_width=True) 
page = st.sidebar.radio("Navigation", ["Map of Ames", "City Sectors", "House Features", 
                        "Feature Engineering", "Renovation Model", "Collaborators"]) 

#=======================================================================================================
# Sidebar House Selector
with st.sidebar.container():
    st.sidebar.title('Model House')

    sec_select = st.sidebar.selectbox('Select Sector',['Downtown','South','West','South East','North','North West'])
    sec_mapper = {'Downtown':'DT','South':'SO','West':'WE','South East':'SE','North':'NO','North West':'NW'}
    model_sec = sec_mapper[sec_select]
    model_neib = st.sidebar.radio('Select Neighborhood',map_data.loc[map_data.Sector==model_sec]['Neighborhood'].unique())

    address_df = map_data.loc[(map_data['Sector']==model_sec) & 
        (map_data['Neighborhood']==model_neib)]

    st.sidebar.markdown(f"### {neib_fullname[model_neib]}")
    try:
        st.sidebar.markdown(f"1-story house median size: *{num_format(basehouse_medians.loc[(model_neib,'1Fl')]['GoodLivArea'])}* sf \
                        \n 1-story house median price: *${num_format(basehouse_medians.loc[(model_neib,'1Fl')]['SalePrice'])}*")
    except: pass
    try:
        st.sidebar.markdown(f"2-story house median size: *{num_format(basehouse_medians.loc[(model_neib,'2Fl')]['GoodLivArea'])}* sf \
                        \n 2-story house median price: *${num_format(basehouse_medians.loc[(model_neib,'2Fl')]['SalePrice'])}*")
    except: pass

#------------------------------------------------------------------------------------------------------
# Page1: Map of Ames, IA
if page == "Map of Ames":
    with st.container():
        st.title('Map of Ames')
        col1, col2 = st.columns([3, 1]) #Set Columns

        # Sidebar Radio Button
        # For selecting map plot
        map_choice = col2.radio("Choose Map:", ('SalePrice', 'Neighborhood', 'Sector'))

        def bok_layer(map_choice = map_choice, fig = bok_fig()):
            # Set map data, hover tool, and color palette
            if map_choice == 'SalePrice':
                mycolors = linear_cmap(field_name='SalePrice', palette=Spectral11, low=min(map_data.SalePrice) ,high=max(map_data.SalePrice))
                color_bar = ColorBar(color_mapper=mycolors['transform'], width=8,  location=(0,0),title="Price $(thousands)")
                fig.add_layout(color_bar, 'right')
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('Price', '@SalePrice')]
                fig.add_tools(my_hover)
                
            elif map_choice == 'Neighborhood':
                mycolors = linear_cmap(field_name='le_Neighbor', palette=Spectral11, low=min(map_data.le_Neighbor) ,high=max(map_data.le_Neighbor))
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('', '@Neighborhood')]
                fig.add_tools(my_hover)
            else:
                mycolors = linear_cmap(field_name='le_Sector', palette=Spectral11, low=min(map_data.le_Sector) ,high=max(map_data.le_Sector))    
                my_hover = HoverTool(names=['House'])
                my_hover.tooltips = [('', '@Neighborhood')]
                fig.add_tools(my_hover)

            # Dots for Houses
            fig.circle(x="x_merc", y="y_merc",
                    size=7,
                    fill_color=mycolors, line_color='black', line_width=0.5,
                    fill_alpha=0.8,
                    name='House',
                    source=map_data)
            

            # Big Dots for Landmarks, with Hover interactivity
            my_hover = HoverTool(names=['landmark'])
            my_hover.tooltips = [('', '@landmarks')]
            fig.circle(x="x_merc", y="y_merc",
                    size=18,
                    fill_color="pink", line_color='red',
                    fill_alpha=0.8,
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

        with col1.expander("K-means Classifier results"):
            st.write("""
                Elbow Plot and Set K=6
            """)
            st.image(filepath+'/assets/K_means_elbow.png')
            st.image(filepath+'/assets/K_means_map.png')

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

    if 'category_order' not in st.session_state :
        st.session_state.category_order = ['Fair','Typical', 'Good','Excellent']

    data_load_state = st.text('Loading data...')
    pick = st.selectbox(
         'Select a feature:',
         ('KitchenQual','BsmtCond','GarageQual', 'PavedDrive', 'CentralAir', 'HeatingQC',))

    if pick == 'KitchenQual' :
        st.session_state.category_order = ['Fair','Typical', 'Good','Excellent']
    if pick == 'HeatingQC' :
        st.session_state.category_order = ['Fair','Typical', 'Good','Excellent']
    elif pick == 'GarageQual': 
        st.session_state.category_order = ['No Garage', 'Fair', 'Typical', 'Good']
    elif pick == 'BsmtCond' : 
        st.session_state.category_order = ['No Basement', 'Fair', 'Typical', 'Good']
    elif pick == 'PavedDrive' :
        st.session_state.category_order = ['N', 'Y']
    elif pick == 'CentralAir' :
        st.session_state.category_order = ['N', 'Y']

    # st.write(st.session_state)
    fig = px.scatter(page_3_data,x='GoodLivArea',y='SalePrice',facet_col=pick,color=pick,trendline='ols',width=900, height=500,
    title = 'Sale Price vs. GoodLivArea by ' + pick, category_orders={pick : st.session_state.category_order})
    st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------
# Page 4 Feature Engineering
elif page == "Feature Engineering":
    st.title('Feature Engineering')


#------------------------------------------------------------------------------------------------------
# Page 6 Modeling
elif page == "Renovation Model":
    with st.container():
        st.title('Renovation Modeler')
        col_main, col_empty, col_b, col_bpx, col_r, col_rpx = st.columns([3,0.3,2,2,2,2]) #Set Columns
        col_main.markdown('##### Select House')
        col_b.markdown('##### Details')
        col_r.markdown('##### Renovation')
    
    with st.container():
        col_main, col_empty, col_b, col_bpx, col_r, col_rpx = st.columns([3,0.3,2,2,2,2]) #Set Columns

        #------Set Base Prediction House---------
        #model_sec = sec_mapper[sec_select]
        #model_neib = col_main.radio('Select Neighborhood',map_data.loc[map_data.Sector==model_sec]['Neighborhood'].unique())
        #pkl_basehouse = pkl_dum_encode(pkl_basehouse, model_neib, 'Neighborhood_')

        #**********************
        with col_main.container():

            address_df = map_data.loc[(map_data['Sector']==model_sec) & 
                (map_data['Neighborhood']==model_neib)]

            source = ColumnDataSource(address_df)
            template = """
                <div style="font-weight: 600; 
                    color: black"> 
                <%= value %>
                </div>
                """
            formatter = HTMLTemplateFormatter(template=template)
            columns = [TableColumn(field="Prop_Addr", title="House Address", formatter=formatter)]

            # define events
            source.selected.js_on_change("indices",
                CustomJS(args=dict(source=source),
                code="""
                document.dispatchEvent(
                new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}})
                )
                """)
                )

            mytable = DataTable(source=source, columns=columns, height=300)
            result = streamlit_bokeh_events(
                bokeh_plot=mytable, 
                events="INDEX_SELECT", 
                key="House", 
                refresh_on_update=True, 
                debounce_time=0,
                override_height=300)

            if result:
                if result.get("INDEX_SELECT"):
                    st.markdown(f'#### **{address_df.iloc[result.get("INDEX_SELECT")["data"],13].values[0]}**')
                    basehouse_PIN = address_df.index.values[result.get("INDEX_SELECT")["data"]][0]
            pkl_basehouse = pkl_data.loc[[basehouse_PIN]]

        hstype_mapper = {1:'Duplex or 2-Family', 2:'2-Story Townhouse', 3:'Split Foyer', 
                        4:'1-Story Townhouse', 5:'1-Story House', 6:'2-Story House'}
        col_main.caption(f"{hstype_mapper[pkl_basehouse['MSSubClass'].values[0]]} in {model_neib}")

        def box_layer(fig = bok_fig(300,260,True)):
            # Set map data, hover tool, and color palette
            mycolors = linear_cmap(field_name='SalePrice', palette=Spectral11, low=min(map_data.SalePrice) ,high=max(map_data.SalePrice))
            #my_hover = HoverTool(names=['House'])
            #my_hover.tooltips = [('Price', '@SalePrice')]
            #fig.add_tools(my_hover)
            # Dots for Houses
            fig.circle(x="x_merc", y="y_merc",
                    size=7,
                    fill_color=mycolors, line_color='black', line_width=0.5,
                    fill_alpha=0.8,
                    name='House',
                    source=address_df)
            fig.xaxis.visible = False
            fig.yaxis.visible = False
            fig.title.visible = False
            return fig
        col_main.bokeh_chart(box_layer())
        #model_hstype = col_main.radio('Select Type of House',map_data.loc[map_data.Neighborhood==model_neib]['MSSubClass'].unique())
        #pkl_basehouse = pkl_dum_encode(pkl_basehouse, model_hstype, 'MSSubClass_')

        # Set & Display Selected Neighborhood and HouseType medians
        
        #pkl_basehouse['GoodLivArea'] = basehouse_medians.loc[(model_neib, model_hstype)]['GoodLivArea']
        #col_main.caption(f"Median SquareFootage: {num_format(pkl_basehouse['GoodLivArea'][0])}")
        #pkl_basehouse['YearBuilt'] = basehouse_medians.loc[(model_neib, model_hstype)]['YearBuilt']
        #col_main.caption(f"Median Year Built: {str(pkl_basehouse['YearBuilt'][0])}")
        #pkl_basehouse['PorchArea'] = basehouse_medians.loc[(model_neib, model_hstype)]['PorchArea']
        #pkl_basehouse['GarageCars'] = 1
        Qual_mapper = {1: 'Fair',2: 'Average', 3: 'Good', 4: 'Excellent'}

        pkl_renohouse = pkl_basehouse.copy()
        # HOUSE RENO Details
        # Above Ground Bathrooms
        col_b.markdown(f"Bathrooms (abv ground): **{num_format(pkl_basehouse['AllBathAbv'].values[0])}**")
        reno_AGbaths = col_r.slider('Build Bathrooms', 0.0, 2.0, 0.0, 0.5)
        pkl_renohouse['AllBathAbv'] = pkl_basehouse['AllBathAbv'].values[0] + reno_AGbaths

        # Kitchen Quality
        col_b.markdown(f"Kitchen Quality: **{Qual_mapper[pkl_basehouse['KitchenQual'].values[0]]}**")
        reno_Kitchen = col_r.radio('Remodel Kitchen',['No', 'Yes'])
        if reno_Kitchen == 'Yes':
            pkl_renohouse['KitchenQual'] = 4

        # Basement Condition
        col_b.markdown(f"Basement Condition: **{Qual_mapper[pkl_basehouse['BsmtCond'].values[0]]}**")
        reno_Bsmt = col_r.radio('Remodel Basement',['No', 'Yes'])
        if reno_Bsmt == 'Yes':
            pkl_renohouse['BsmtCond'] = 4 
        reno_FinBsmt = col_r.radio('Finish Basement',['No', 'Yes'])
        if reno_FinBsmt == 'Yes':
            pkl_renohouse['GoodLivArea'] = pkl_renohouse['GoodLivArea'] + pkl_renohouse['BsmtUnfSF']
            pkl_renohouse['BsmtUnfSF'] = 0

        # Garage Quality
        col_b.markdown(f"Garage Quality: **{Qual_mapper[pkl_basehouse['GarageQual'].values[0]]}**")
        reno_Garage = col_r.radio('Remodel Garage',['No', 'Yes'])
        if reno_Garage == 'Yes':
            pkl_renohouse['GarageQual'] = 4 

        # Pool
        if pkl_basehouse['HasPool'].values[0] == 0:
            base_pool = col_b.radio('Pool',['No'])
            reno_pool = col_r.radio('Build Pool',['No', 'Yes'])
            pkl_renohouse['HasPool'] = 0 if reno_pool == 'No' else 1
        else:
            base_pool = col_b.radio('Pool',['Yes'])
        
        # Central Air
        if pkl_basehouse['CentralAir'].values[0] == 0:
            base_cAir = col_b.radio('Central Air',['No'])
            reno_cAir = col_r.radio('Install Central Air',['No', 'Yes'])
            pkl_renohouse['CentralAir_Y'] = 0 if reno_cAir == 'No' else 1
        else:
            base_cAir = col_b.radio('Central Air',['Yes'])
        
        # Paved Driveway
        if pkl_basehouse['PavedDrive'].values[0] == 0:
            base_pave = col_b.radio('Paved Driveway',['No'])
            reno_pave = col_r.radio('Pave Driveway',['No', 'Yes'])
            pkl_renohouse['PavedDrive_Y'] = 0 if reno_pave == 'No' else 1
        else:
            base_pave = col_b.radio('Paved Driveway',['Yes'])

        # Base House MODEL PRICE
        pkl_baseprice = np.floor(pkl_model.predict(pkl_basehouse)[0])
        col_bpx.subheader(f'**${num_format(pkl_baseprice)}**')
        col_bpx.caption('Baseline Price Prediction')
        col_bpx.write('-------------------------')
        col_bpx.caption(f"Actual Price: **${num_format(pkl_basehouse['SalePrice'].values[0])}**")
        col_bpx.markdown(f"Livable Space: **{num_format(pkl_basehouse['GoodLivArea'].values[0])}** sf")
        col_bpx.markdown(f"Unfinished Bsmt: **{num_format(pkl_basehouse['BsmtUnfSF'].values[0])}** sf")
        col_bpx.markdown(f"Garage Size: **{num_format(pkl_basehouse['GarageCars'].values[0])}** cars")
        col_bpx.markdown(f"Porch or Deck: **{num_format(pkl_basehouse['PorchArea'].values[0])}** sf")
        
        # Renovated House PRICE
        pkl_renoprice = np.floor(pkl_model.predict(pkl_renohouse)[0])
        col_rpx.subheader(f'**${num_format(pkl_renoprice)}**')
        col_rpx.caption('Renovated House Price')

        # Added metric
        percent_change = round((((pkl_renoprice - pkl_baseprice)/pkl_baseprice)*100),2)
        # col_rpx.markdown(f'### **${num_format(pkl_renoprice - pkl_baseprice)}**')
        col_rpx.metric(label='',value='${0}'.format(num_format(pkl_renoprice - pkl_baseprice)),delta='{0}%'.format(percent_change))
        col_rpx.caption('Difference')

#------------------------------------------------------------------------------------------------------
# Page 7 About Page
elif page == "Collaborators":
    st.title('Collaborators')
    with st.container():
        col1, col2, col3 = st.columns([2,1,7])
        col1.subheader('Daniel Nie')
        col2.markdown('#### [Github](https://github.com/dnie44)')
        col3.markdown('#### [LinkedIn](https://www.linkedin.com/in/danielnie/)')
        col1.subheader('David Kressley')
        col2.markdown('#### [Github](https://github.com/Skipp-py)')
        col3.markdown('#### [LinkedIn](https://www.linkedin.com/in/david-kressley-2a4a2194/)')
        col1.subheader('Karl Lundquist')
        col2.markdown('#### [Github](https://github.com/klundquist)')
        col3.markdown('#### [LinkedIn](https://www.linkedin.com/in/karl-lundquist/)')
        col1.subheader('Tony Pennoyer')
        col2.markdown('#### [Github](https://github.com/tonypennoyer)')
        col3.markdown('#### [LinkedIn](https://www.linkedin.com/in/tony-pennoyer-155172123/)')
    with st.container():
        col1, col2 = st.columns([4,6])
        col1.info('We are Machine Learning Fellows at [NYC Data Science Academy]\
                (https://nycdatascience.com/)')
        col1.caption('Updated: 11/26/2021')