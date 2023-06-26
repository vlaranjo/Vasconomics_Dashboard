###***************** Import the Libraries ##########################

import streamlit as st

import pandas as pd
import numpy as np

import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay, CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

from yahooquery import Ticker
import talib as tb

import holoviews as hv
from holoviews import opts, dim
from bokeh.models import HoverTool

hv.extension('bokeh')

# ********** Define the Frequencies and the Times **********
# Define the frequencies and their corresponding time differences in days
frequencies_table = {'1D': 1, '1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '2Y': 504, '3Y': 756, '5Y': 1260, '10Y': 2520}

frequencies_graph = {'3M': 63,'1W': 5, '1M': 21, '6M': 126, '1Y': 252, '2Y': 504, '3Y': 756, '5Y': 1260, '10Y': 2520}

#Define today and last business day
today = datetime.datetime.today().date()
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
last_bd = (today - us_bd - BDay(1)).to_pydatetime().date() if today.weekday() >= 5 else today

# ********************************************************************************************************************************
# ********************************************************************************************************************************

# ********** Define the Function for the Performance Heatmap Generation **********
@st.cache_resource
def generate_heatmap(indices_list, indices_name_mapping, frequencies, last_bd):
    
    #Revert the order of the index list because it will then be reverted again in the heatmap
    indices_list = indices_list[::-1]
    
    returns_dict = {}  # Dictionary to store the returns for each frequency
    
    #Add progress bar
    progress_bar = st.progress(0, text="Running")

    for index in indices_list:
        yf_symbol_obj = Ticker(index)
        historical_prices = yf_symbol_obj.history(period='max')
        historical_prices = historical_prices.reset_index()
        historical_prices = historical_prices[['date', 'adjclose']]

        returns_list = []  # List to store the returns for each index

        for frequency, num_days in frequencies.items():
            try:
                start_date = (last_bd - num_days * us_bd).to_pydatetime().date()
                filtered_prices = historical_prices.loc[
                    pd.to_datetime(historical_prices['date']) >= pd.to_datetime(start_date)
                ]
                last_price = filtered_prices['adjclose'].iloc[-1]
                start_price = filtered_prices['adjclose'].iloc[0]
                freq_price_return = round(((last_price / start_price - 1) * 100), 2)
            except IndexError:
                freq_price_return = np.nan

            returns_list.append(freq_price_return)

        returns_dict[index] = returns_list

        progress_bar.progress(((indices_list.index(index) + 1) / len(indices_list)), text="Running")

    df = pd.DataFrame(returns_dict)
    df.index = list(frequencies.keys())
    df = df.T
    df.index = df.index.map(indices_name_mapping)

    melted_df = df.reset_index().melt(id_vars='index', var_name='Frequency', value_name='Return')
    melted_df = melted_df[['Frequency', 'index', 'Return']]

    grid_style = {'grid_line_color': 'black', 'grid_line_width': 100}

    def hook(plot, element):
        plot.state.outline_line_width = 2
        plot.state.outline_line_color = "black"

    heatmap_rows_list = list(melted_df['Frequency'].unique())

    heatmaps = []

    for heatmap_row in heatmap_rows_list:
        data = melted_df[melted_df['Frequency'] == heatmap_row]
        heatmap = hv.HeatMap(data, label=f"Frequency {heatmap_row}")
        heatmap = heatmap.opts(
            opts.HeatMap(width=700, height=115, xrotation=45, xaxis='top', labelled=[],
                         tools=['hover'], cmap='RdYlGn',
                         fontsize={'title': 15, 'xticks': 10, 'yticks': 10},
                         ))
        heatmap = heatmap.opts(gridstyle=grid_style, show_grid=True, hooks=[hook])
        heatmap = heatmap * hv.Labels(heatmap).opts(padding=0)

        heatmaps.append(heatmap)

    indices_heatmap = hv.Overlay(heatmaps).opts(opts.Overlay(show_legend=False, height=400))
    
    #Deploy the Holoviews figure
    st.write(hv.render(indices_heatmap, backend='bokeh'))

    start_date_text = datetime.datetime.strftime(start_date, "%m/%d/%Y")
    last_date_text = datetime.datetime.strftime(last_bd, "%m/%d/%Y")
    st.write("Source: Yahoo Finance. Period: " + start_date_text + " - " + last_date_text + ". Past performance is not indicative of future results.")

    return

# ********** Define the Function for the Multi-Perfomance Chart Generation **********
@st.cache_resource(experimental_allow_widgets=True)
def generate_multiperformance_chart(indices_list, indices_name_mapping, frequencies, last_bd, widget_name):

    #Download the Price Data from Yahoo Finance
    #Create an empty dataframe and then run the loop to download the prices for all the indices
    merged_prices_df = pd.DataFrame()
    
    #Add progress bar
    progress_bar = st.progress(0, text="Running")

    #Run the Loop to download the Price data for each index
    for index in indices_list:
        yf_symbol_obj = Ticker(index)
        historical_prices = yf_symbol_obj.history(period='max')
        historical_prices = historical_prices.reset_index()
        historical_prices = historical_prices[['date', 'adjclose']]
        historical_prices = historical_prices.rename(columns={'adjclose': indices_name_mapping[index]})
        if merged_prices_df.empty:
            merged_prices_df = historical_prices
        else:
            merged_prices_df = pd.merge(merged_prices_df, historical_prices, on='date', how='outer')
            
        progress_bar.progress(((indices_list.index(index) + 1) / len(indices_list)), text="Running")

    #Sort the Merge Dataframe from Date ascending and then fill na's with Forward Fill method
    merged_prices_df = merged_prices_df.sort_values('date', ascending=True)
    merged_prices_df = merged_prices_df.fillna(method='ffill')

    #Set column 'Date' as index
    merged_prices_df = merged_prices_df.set_index('date')

    #To Create the Chart with the Dropdown list, will need to create a Dictionary
    ##Explanation: Need to Join the Datrames, which include the perforance data for each index for each frequency, in a Dictionary (3 Dimensions: date, index, and frequency) so that then there can be applied a list including each timeperiod
    multi_index_performance_dict = {}

    for frequency, num_days in frequencies.items():
        #Start Date based on each Frequency
        start_date = (last_bd - num_days * us_bd).to_pydatetime().date()

        #Filter/Slice the Dataframe based on the Frequency and Number of Days
        filtered_merged_prices = merged_prices_df.loc[pd.to_datetime(merged_prices_df.index) >= pd.to_datetime(start_date)]

        #Calculate the Performance with base Index = 100
        filtered_merged_performance = filtered_merged_prices.pct_change().fillna(0)
        filtered_merged_performance.iloc[0] = 0
        filtered_merged_performance = ((1 + filtered_merged_performance).cumprod() * 100).round(2)

        #Add the Filtered/Sliced Dataframe to the Dictionary
        multi_index_performance_dict[frequency] = filtered_merged_performance

    #Function for Chart Generation - Generate all curves
    def load_timeperiods(Timeperiod):
        def getCurves(n):
            for column in multi_index_performance_dict[Timeperiod].columns:
                curve = hv.Curve(multi_index_performance_dict[Timeperiod][column], label = column).opts(
                    tools=[HoverTool(tooltips=[('date', '@{date}{%F}'),(str(column),'@{'+str(column)+'}')], 
                                     formatters={'@{date}': 'datetime','@{'+str(column)+'}':'numeral'})])
                yield curve

        source_curves, target_curves  = [], []
        for curve in getCurves(10):
            tgt = curve.opts(opts.Curve(width=700))
            target_curves.append(tgt)     

        #Overlay the source and target curves
        overlaid_plot_tgt = hv.Overlay(target_curves).relabel('').opts(ylabel='Index', legend_position='left', 
                                                                       background_fill_alpha  = 0, height=400,
                                                                      )

        return overlaid_plot_tgt

    timperiods_names = list(multi_index_performance_dict.keys())

    dmap = hv.DynamicMap(load_timeperiods, kdims='Timeperiod').redim.values(Timeperiod=timperiods_names)

    dmap = dmap.opts(framewise=True).map(lambda obj: obj.opts(framewise=True), hv.Element)
    
    # Get the selected time period from the dropdown list
    selected_timeperiod = st.selectbox(widget_name, timperiods_names, index=0, key=widget_name)
    
    #Deploy the Holoviews figure
    st.write(hv.render(dmap[selected_timeperiod], backend='bokeh'), use_container_width=True)

    start_date_text = datetime.datetime.strftime(start_date, "%m/%d/%Y")
    last_date_text = datetime.datetime.strftime(last_bd, "%m/%d/%Y")
    st.write("Source: Yahoo Finance. Period: " + start_date_text + " - " + last_date_text + ". Past performance is not indicative of future results.")

    return

# ********** Define the Function for the Technical Analysis Chart Generation **********
@st.cache_resource(experimental_allow_widgets=True)
def generate_technical_analysis_chart(indices_list, indices_name_mapping, frequencies, last_bd, widget_name):
    multi_index_ta_dict = {}
    
    #Add progress bar
    progress_bar = st.progress(0, text="Running")

    #Run the Loop to download the Price data for each index
    for index in indices_list:
        yf_symbol_obj = Ticker(index)
        historical_prices = yf_symbol_obj.history(period='max')
        historical_prices = historical_prices.reset_index()
        historical_prices = historical_prices[['date', 'adjclose']]
        index_name = indices_name_mapping[index]
        historical_prices = historical_prices.rename(columns={'adjclose': index_name})

        #Calculate the Moving Averages and the RSI
        historical_prices['200day MA'] = tb.MA(historical_prices[index_name], timeperiod=200, matype=0)
        historical_prices['50day MA'] = tb.MA(historical_prices[index_name], timeperiod=50, matype=0)
        historical_prices['14day RSI'] = tb.RSI(historical_prices[index_name], timeperiod=14)
        historical_prices['70 line'] = 70
        historical_prices['30 line'] = 30

        #Filter data for one year
        num_days = frequencies['1Y']
        start_date = (last_bd - num_days * us_bd).to_pydatetime().date()
        filtered_prices = historical_prices.loc[pd.to_datetime(historical_prices['date']) >= pd.to_datetime(start_date)]
        filtered_prices = filtered_prices.set_index('date')
        multi_index_ta_dict[index_name] = filtered_prices
        
        progress_bar.progress(((indices_list.index(index) + 1) / len(indices_list)), text="Running")

    #Define the function for the graph creation -> Function will be used in the hv.DynamicMap -> Moving Averages
    def load_index_ma(index_name):
        #Creating the Moving Averages Graph
        ma_graph_ticker_line = hv.Curve(multi_index_ta_dict[index_name][index_name], label = index_name).opts(tools = ['hover'], color = 'black', ylabel='Price', width = 800, height = 350)
        ma_graph_50d_line = hv.Curve(multi_index_ta_dict[index_name]['50day MA'], label = '50day MA').opts(tools = ['hover'], color = 'gold', ylabel='Price', width = 800, height = 350)
        ma_graph_200d_line = hv.Curve(multi_index_ta_dict[index_name]['200day MA'], label = '200day MA').opts(tools = ['hover'], color = 'darkmagenta', ylabel='Price', width = 800, height = 350)
        ma_graph = hv.Overlay(ma_graph_ticker_line * ma_graph_50d_line * ma_graph_200d_line).opts(legend_position='top', title = index_name + ': Moving Averages and RSI')

        return ma_graph

    #Define the function for the graph creation -> Function will be used in the hv.DynamicMap -> RSI
    def load_index_rsi(index_name):    
            #Creating the RSI Graph
        rsi_line = hv.Curve(multi_index_ta_dict[index_name]['14day RSI']).opts(tools = ['hover'], color = 'black', ylabel='RSI', width = 800, height = 250)
        rsi_70_line = hv.Curve(multi_index_ta_dict[index_name]['70 line']).opts(tools = ['hover'], color = 'red', ylabel='RSI', width = 800, height = 250)
        rsi_30_line = hv.Curve(multi_index_ta_dict[index_name]['30 line']).opts(tools = ['hover'], color = 'green', ylabel='RSI', width = 800, height = 250)
        rsi_graph = hv.Overlay(rsi_line * rsi_70_line * rsi_30_line)
        
        return rsi_graph

    #Define the Indices list to be used in the DynamicMap dropdown list
    index_names = list(multi_index_ta_dict.keys())

    #Instantiate the DynamicMap object - for MA
    dmap_ma = hv.DynamicMap(load_index_ma, kdims='index_name').redim.values(index_name=index_names)
    dmap_ma = dmap_ma.opts(framewise=True).map(lambda obj: obj.opts(framewise=True), hv.Element)
    
    #Instantiate the DynamicMap object - for MA
    dmap_rsi = hv.DynamicMap(load_index_rsi, kdims='index_name').redim.values(index_name=index_names)
    dmap_rsi = dmap_rsi.opts(framewise=True).map(lambda obj: obj.opts(framewise=True), hv.Element)
    
    # Get the selected time period from the dropdown list
    selected_index= st.selectbox(widget_name, index_names, index=0, key=widget_name)
    
    #Deploy the Holoviews figures - MA and RSI
    st.write(hv.render(dmap_ma[selected_index], backend='bokeh'), use_container_width=True)
    
    st.write(hv.render(dmap_rsi[selected_index], backend='bokeh'), use_container_width=True)
    
    start_date = multi_index_ta_dict[index_name].index[0]
    start_date_text = datetime.datetime.strftime(start_date, "%m/%d/%Y")
    last_date_text = datetime.datetime.strftime(last_bd, "%m/%d/%Y")
    st.write("Source: Yahoo Finance. Period: " + start_date_text + " - " + last_date_text + ". Past performance is not indicative of future results.")

    return

# ********************************************************************************************************************************
# ********************************************************************************************************************************

#********** Start of the Dashboard **********

st.set_page_config(page_title="Commodities", page_icon="🛢️")

#Title of the Application
st.title("Commodities")

#Sub-title of the Application
st.write("This section presents an overview of commodities performance, including Sectors and Components")

# Existing Streamlit page content
st.sidebar.header("Commodity Sections")

#********** Commodity Sectors Section **********
st.header("Commodity - Sectors")
st.sidebar.markdown('<a href="#commodity-sectors" style="color: black;">- Commodity - Sectors</a>', unsafe_allow_html=True)

#Define the List of Commodity Sectors Indices
energy_index = 'AIGE.MI'
aggriculture_index = 'AIGA.MI'
industrial_metals_index = 'AIGI.MI'
precious_metals_index = 'AIGP.MI'
softs_index = 'AIGS.MI'

#Create an object with the list of objects
commodity_sectors_indices = [energy_index, aggriculture_index, industrial_metals_index, precious_metals_index, softs_index]

#Name Mapping 
commodity_sectors_index_name_mapping = {'AIGE.MI': 'Energy', 'AIGA.MI':'Aggriculture', 'AIGI.MI':'Industrial Metals', 
                                    'AIGP.MI': 'Precious Metals', 'AIGS.MI': 'Softs'}

#Commodity Sectors Indices - Performance Table
st.subheader("Commodity - Sectors Performance Table (%)")

st.write("Note: This subsection uses ETFs for performance. A table with the ETF corresponding to each label is presented at the bottom of the page")

generate_heatmap(indices_list = commodity_sectors_indices, indices_name_mapping = commodity_sectors_index_name_mapping, frequencies = frequencies_table, last_bd = last_bd)

#Commodity Sectors Indices - Performance Chart
st.subheader("Commodity - Sectors Technical Analysis Chart (%)")

st.write("Note: This subsection uses ETFs for performance. A table with the ETF corresponding to each label is presented at the bottom of the page")

generate_technical_analysis_chart(indices_list = commodity_sectors_indices, indices_name_mapping = commodity_sectors_index_name_mapping, frequencies = frequencies_graph, last_bd = last_bd, widget_name = 'Select Sector, Commodity Sectors')

#********** Commodity Components Section **********
st.header("Commodity - Components")
st.sidebar.markdown('<a href="#commodity-components" style="color: black;">- Commodity - Components</a>', unsafe_allow_html=True)

#Define the List of Equity Regions Indices
crude_oil_index = 'BRNT.MI'
nat_gas_index = 'NGAS.MI'
corn_index = 'CORN.MI'
soybeans_index = 'SOYB.MI'
wheat_index = 'WEAT.MI'
gold_index = 'XAD5.MI'
silver_index = 'PHAG.AS'
platinum_index = 'PHPT.MI'
copper_index = 'COPA.MI'
aluminum_index = 'ALUM.MI'
zinc_index = 'ZINC.MI'
nickel_index = 'NICK.MI'
sugar_index = 'SUGA.MI'
coffee_index = 'COFF.MI'
cotton_index = 'COTN.MI'

#Create an object with the list of objects
commodity_components_indices = [crude_oil_index, nat_gas_index, corn_index, soybeans_index, wheat_index, gold_index, silver_index, platinum_index, copper_index, aluminum_index, zinc_index, nickel_index, sugar_index, coffee_index, cotton_index]

#Name Mapping 
commodity_components_index_name_mapping = {'BRNT.MI': 'Brent Crude Oil','NGAS.MI': 'Natural Gas','CORN.MI': 'Corn',
                                     'SOYB.MI': 'Soybeans','WEAT.MI': 'Wheat ','XAD5.MI': 'Gold','PHAG.AS': 'Silver',
                                     'PHPT.MI': 'Platinum','COPA.MI': 'Copper','ALUM.MI': 'Aluminium','ZINC.MI': 'Zinc ',
                                     'NICK.MI': 'Nickel','SUGA.MI': 'Sugar','COFF.MI': 'Coffee','COTN.MI': ' Cotton '}

#Commodity Components Indices - Performance Table
st.subheader("Commodity - Components Performance Table (%)")

generate_heatmap(indices_list = commodity_components_indices, indices_name_mapping = commodity_components_index_name_mapping, frequencies = frequencies_table, last_bd = last_bd)

#Commodity Components Indices - Performance Chart
st.subheader("Commodity - Components Technical Analysis Chart (%)")

generate_technical_analysis_chart(indices_list = commodity_components_indices, indices_name_mapping = commodity_components_index_name_mapping, frequencies = frequencies_graph, last_bd = last_bd, widget_name = 'Select Component, Commodity Components')

#Table with ETFs corresponding to each label
#Equity Regions - ETF Table
st.subheader("Equities - Countries - ETF Table")

commodity_sectors_etf_table = """
| Label             | ETF Name                      | ETF Ticker |
|-------------------|-------------------------------|------------|
| Energy            | WisdomTree Energy             | AIGE       |
| Aggriculture      |  WisdomTree   Agriculture     | AIGA       |
| Industrial metals | WisdomTree Industrial Metals  | AIGI       |
| Precious metals   | WisdomTree Precious Metals    | AIGP       |
| Softs             | WisdomTree Softs              | AIGS       |
"""
    
st.write(commodity_sectors_etf_table)

#Commodity Components - ETF Table
st.subheader(" ")
st.subheader("Commodity - Components - ETF Table")

commodity_components_etf_table = """
| Label       | ETF Name                     | ETF Ticker |
|-------------|------------------------------|------------|
| Crude oil   | WisdomTree Brent Crude Oil   | BRNT       |
| Natural gas | WisdomTree Natural Gas       | NGAS       |
| Corn        | WisdomTree Corn              | CORN       |
| Soybeans    | WisdomTree Soybeans          | SOYB       |
| Wheat       | WisdomTree Wheat             | WEAT       |
| Gold        | Xtrackers Physical Gold ETC  | XAD5       |
| Silver      | WisdomTree Physical Silver   | PHAG       |
| Platinum    | WisdomTree Physical Platinum | PHPT       |
| Copper      | WisdomTree Copper            | COPA       |
| Aluminum    | WisdomTree Aluminium         | ALUM       |
| Zinc        | WisdomTree Zinc              | ZINC       |
| Nickel      | WisdomTree Nickel            | NICK       |
| Sugar       | WisdomTree Sugar             | SUGA       |
| Coffee      | WisdomTree Coffee            | COFF       |
| Cotton      |  WisdomTree Cotton           | COTN       |
"""
    
st.write(commodity_components_etf_table)

#****************************************************************************************************************************************
# Add Linkedin

linkedin_url = "https://www.linkedin.com/in/vascolaranjo/"

linkedin_html = """
    <style>
        #linkedin-symbol {{
            position: absolute;
            top: 10px;
            left: 10px;
            display: flex;
            align-items: center;
        }}
        .contact-text {{
            margin-left: 5px;
        }}
    </style>
    <div id="linkedin-symbol">
        <a href="{url}" target="_blank">
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"
                alt="LinkedIn" height="30" width="30">
        </a>
    <div class="contact-text">Contact me</div>
    </div>
"""

st.sidebar.markdown(linkedin_html.format(url=linkedin_url), unsafe_allow_html=True)