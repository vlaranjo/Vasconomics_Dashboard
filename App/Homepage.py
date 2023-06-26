###***************** Import the Libraries ##########################

import streamlit as st

st.set_page_config(page_title="Homepage", page_icon="👋")

st.write("# Welcome to Vasconomics Macro Dashboard! 👋")

st.markdown("This is a Dashboard for Global Financial Markets, including Equities, Fixed Income, Commodities, and more")

st.markdown("The sections currently included are:")

st.markdown('<a href="/Asset_Classes" style="color: black;">🌐 Asset Classes</a>', unsafe_allow_html=True)
st.markdown('<a href="/Equities" style="color: black;">📈 Equities</a>', unsafe_allow_html=True)
st.markdown('<a href="/US_Majors_&_Factors" style="color: black;">- US Equities Majors & Style Factors</a>', unsafe_allow_html=True)
st.markdown('<a href="/US_Sectors" style="color: black;">- US Equities Sectors</a>', unsafe_allow_html=True)
st.markdown('<a href="/Fixed_Income" style="color: black;">💵 Fixed Income</a>', unsafe_allow_html=True)
st.markdown('<a href="/Commodities" style="color: black;">🛢️ Commodities</a>', unsafe_allow_html=True)
st.markdown('<a href="/FX" style="color: black;">💱 FX</a>', unsafe_allow_html=True)
st.markdown('<a href="/Volatility" style="color: black;">🚨 Volatility</a>', unsafe_allow_html=True)
st.markdown('<a href="/Under_Development" style="color: black;">🚧 Under Development</a>', unsafe_allow_html=True)

st.markdown("""
            
            Note: I make my best effort to show performance as total return. I do this by picking "Accumulating" ETFs when available, and only choosing "Distributing" otherwise. Also, I use the Adjusted Close from Yahoo Finance, which adjusts for dividend distributions. Differences arise over time, especially for longer periods, in comparing "Accumulating" with "Distributing" ETFs as the former is a preferred vehicle for the compounding of returns.
            """)

#***************************************************************************************************************************************
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