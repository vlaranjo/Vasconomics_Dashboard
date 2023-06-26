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

# ********************************************************************************************************************************
# ********************************************************************************************************************************

#********** Start of the Dashboard **********
 
st.set_page_config(page_title="Fixed_Income", page_icon="💵")
 
#Title of the Application
st.title("Fixed Income")

#Sub-title of the Application
st.write("This section presents an overview of Fixed Income performance, including ")

# Existing Streamlit page content
st.sidebar.header("Fixed Income Sections")

#********** Equity US Major Indices Section **********
st.header("Fixed Income - Performance")
st.sidebar.markdown('<a href="#fixed-income-performance" style="color: black;">- Fixed Income - Performance</a>', unsafe_allow_html=True)

#Define the List of Equity US Major Indices
us_treasuries_index = 'SYBT.DE'
eur_govies_index = 'VGEA.DE'
us_tips_index = 'IUST.DE'
eur_inf_prot_index = 'IBCI.AS'
uk_gilts_index = 'IGLT.MI'
em_local_bond_index = 'SPFA.DE'
us_IG_bond_index = 'IBCD.DE'
us_HY_bond_index = 'IHYU.MI'
eur_IG_bond_index = 'VECP.DE'
eur_HY_bond_index = 'IHYG.MI'
us_mbs_index = 'QDVP.DE'
us_muni_bond_index = 'MUNI.MI'

#Create an object with the list of objects
fixed_income_indices = [us_treasuries_index, eur_govies_index, us_tips_index, eur_inf_prot_index, uk_gilts_index, em_local_bond_index,
                       us_IG_bond_index, us_HY_bond_index, eur_IG_bond_index, eur_HY_bond_index, us_mbs_index, us_muni_bond_index]

#Name Mapping 
fixed_income_indices_index_name_mapping = {'SYBT.DE': 'US Treasuries', 'VGEA.DE':'EUR Govies', 'IUST.DE':'US TIPS', 
                                           'IBCI.AS': 'EUR Infl. Protected Bonds', 'IGLT.MI': 'UK GILTS','SPFA.DE': 'EM Govies Local CCY',
                                           'IBCD.DE': 'US IG Bonds', 'IHYU.MI': 'US HY Bonds','VECP.DE': 'EUR IG Bonds', 
                                           'IHYG.MI': 'EUR HY Bonds', 'QDVP.DE': 'US MBS', 'MUNI.MI': 'US Muni Bonds'}

#Equity US Major Indices - Performance Table
st.subheader("Fixed Income Performance Table (%)")

st.write("Note: This subsection uses ETFs for performance. A table with the ETF corresponding to each label is presented at the bottom of the page")

generate_heatmap(indices_list = fixed_income_indices, indices_name_mapping = fixed_income_indices_index_name_mapping, frequencies = frequencies_table, last_bd = last_bd)

#***************************************************************************************************************************************







#***************************************************************************************************************************************

#Table with ETFs corresponding to each label
#Equity US Major Indices - ETF Table
st.subheader("Fixed Income - ETF Table")

fixed_income_etf_table = """
| Label                     | ETF Name                                             | ETF Ticker |
|---------------------------|------------------------------------------------------|------------|
| US Treasuries             | SPDR® Bloomberg U.S. Treasury Bond UCITS ETF         | SYBT       |
| EUR Govies                | Vanguard EUR Eurozone Government Bond UCITS ETF      | VGEA       |
| US TIPS                   | iShares $ TIPS UCITS ETF                             | IUST       |
| EUR Infl. Protected Bonds | iShares € Inflation Linked Govt Bond UCITS ETF       | IBCI       |
| UK GILTS                  | iShares Core UK Gilts UCITS ETF                      | IGLT       |
| EM Govies Local CCY       | SPDR Bloomberg Emerging Markets Local Bond UCITS ETF | SPFA       |
| US IG Bonds               | iShares $ Corp Bond UCITS ETF                        | IBCD       |
| US HY Bonds               | iShares $ High Yield Corp Bond UCITS ETF             | IHYU       |
| EUR IG Bonds              | Vanguard EUR Corporate Bond UCITS ETF                | VECP       |
| EUR HY Bonds              | iShares € High Yield Corp Bond UCITS ETF             | IHYG       |
| US MBS                    | iShares US Mortgage Backed Securities UCITS ETF      | QDVP       |
| US Muni Bonds             | Invesco US Municipal Bond UCITS ETF                  | MUNI       |
"""

st.write(fixed_income_etf_table)

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