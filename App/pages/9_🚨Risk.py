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

    indices_heatmap = hv.Overlay(heatmaps).opts(opts.Overlay(show_legend=False, height=300))
    
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
def generate_averages_chart(indices_list, indices_name_mapping, frequencies, last_bd, widget_name):
    multi_index_averages_dict = {}
    
    #Add progress bar
    progress_bar = st.progress(0, text="Running")

    for index in indices_list:
        yf_symbol_obj = Ticker(index)
        historical_prices = yf_symbol_obj.history(period='max')
        historical_prices = historical_prices.reset_index()
        historical_prices = historical_prices[['date', 'close']]
        index_name = indices_name_mapping[index]
        historical_prices = historical_prices.rename(columns={'close': index_name})

        #Calculate the Types of Averages
        start_date_1Y = (last_bd - frequencies['1Y'] * us_bd).to_pydatetime().date()
        filtered_prices_1Y = historical_prices.loc[pd.to_datetime(historical_prices['date']) >= pd.to_datetime(start_date_1Y)]
        start_date_10Y = (last_bd - frequencies['10Y'] * us_bd).to_pydatetime().date()
        filtered_prices_10Y = historical_prices.loc[pd.to_datetime(historical_prices['date']) >= pd.to_datetime(start_date_10Y)]

        historical_prices['1Year Avg.'] = filtered_prices_1Y.mean().iloc[0]
        historical_prices['10Year Avg.'] = filtered_prices_10Y.mean().iloc[0]
        historical_prices['Historical Avg.'] = historical_prices.mean().iloc[0]

        #Set Index as 'date'
        historical_prices = historical_prices.set_index('date')

        #Join Index data in the dictionary
        multi_index_averages_dict[index_name] = historical_prices
        
        progress_bar.progress(((indices_list.index(index) + 1) / len(indices_list)), text="Running")

    def load_index_averages_graph(index_name):
        #Creating the Moving Averages Graph
        avg_graph_ticker_line = hv.Curve(multi_index_averages_dict[index_name][index_name], label = index_name).opts(tools = ['hover'], color = 'black', ylabel='Price', width = 800, height = 350)
        avg_graph_1Y_avg_line = hv.Curve(multi_index_averages_dict[index_name]['1Year Avg.'], label = '1Year Avg.').opts(tools = ['hover'], color = 'blueviolet', line_dash='dashed', ylabel='Price', width = 800, height = 350)
        avg_graph_10Y_avg_line = hv.Curve(multi_index_averages_dict[index_name]['10Year Avg.'], label = '10Year Avg.').opts(tools = ['hover'], color = 'orangered', line_dash='dashed', ylabel='Price', width = 800, height = 350)
        avg_graph_hist_avg_line = hv.Curve(multi_index_averages_dict[index_name]['Historical Avg.'], label = 'Historical Avg.').opts(tools = ['hover'], color = 'limegreen', line_dash='dashed', ylabel='Price', width = 800, height = 350)
        avg_graph = hv.Overlay(avg_graph_ticker_line * avg_graph_1Y_avg_line * avg_graph_10Y_avg_line * avg_graph_hist_avg_line).opts(legend_position='top', title = index_name + ': History and Historical Averages')

        return avg_graph

    #Define the Indices list to be used in the DynamicMap dropdown list
    index_names = list(multi_index_averages_dict.keys())

    #Instantiate the DynamicMap object - for MA
    dmap_avg = hv.DynamicMap(load_index_averages_graph, kdims='index_name').redim.values(index_name=index_names)
    dmap_avg = dmap_avg.opts(framewise=True).map(lambda obj: obj.opts(framewise=True), hv.Element)

    # Get the selected time period from the dropdown list
    selected_index= st.selectbox(widget_name, index_names, index=0, key=widget_name)

    #Deploy the Holoviews figures - Averages Graph
    st.write(hv.render(dmap_avg[selected_index], backend='bokeh'), use_container_width=True)

    start_date = multi_index_averages_dict[index_name].index[0]
    start_date_text = datetime.datetime.strftime(start_date, "%m/%d/%Y")
    last_date_text = datetime.datetime.strftime(last_bd, "%m/%d/%Y")
    st.write("Source: Yahoo Finance. Period: " + start_date_text + " - " + last_date_text + ". Past performance is not indicative of future results.")
    
    return

# ********************************************************************************************************************************
# ********************************************************************************************************************************

#********** Start of the Dashboard **********

st.set_page_config(page_title="Volatility", page_icon="🚨")

#Title of the Application
st.title("Volatility")

#Sub-title of the Application
st.write("This section presents an overview of Volatility Indices, including VIX, SKEW and MOVE")

# Existing Streamlit page content
st.sidebar.header("Volatility Sections")

#********** Volatility - Historically Section **********
st.header("Volatility - Historically")
st.sidebar.markdown('<a href="#volatility-historically" style="color: black;">- Volatility - Historically</a>', unsafe_allow_html=True)

#Define the List of FX Indices
vix_index = '^VIX'
skew_index = '^SKEW'
move_index = '^MOVE'

#Create an object with the list of objects
volatility_indices = [vix_index, skew_index, move_index]

#Name Mapping 
volatility_indices_name_mapping = {'^VIX': 'CBOE Volatility Index (VIX)', '^SKEW': 'CBOE SKEW Index', '^MOVE': 'ICE BofAML MOVE Index'}

#Volatility - Historically - Performance Table
st.subheader("Volatility - Performance Table (%)")

st.write("Note: This subsection uses the indices for performance. Use the Box Zoom tool in the options on the right-hand side for zooming in dates.")

generate_heatmap(indices_list = volatility_indices, indices_name_mapping = volatility_indices_name_mapping, frequencies = frequencies_table, last_bd = last_bd)

#Volatility - Historically - Averages Chart
st.subheader("Volatility - History Chart (%)")

st.write("Note: This subsection uses the indices for performance.")

generate_averages_chart(indices_list = volatility_indices, indices_name_mapping = volatility_indices_name_mapping, frequencies = frequencies_graph, last_bd = last_bd, widget_name = 'Select Index, Volatility Indices')

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