# -*- coding: utf-8 -*-
import io
import math
import os

import branca
import folium
import matplotlib as mpl
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import mplstereonet

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from folium.plugins import Draw
from haversine import haversine

from scipy.optimize import least_squares, minimize
from streamlit_folium import st_folium







def fs(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.sin(math.radians(ldeg))**lnn, deg, nnn)))


def fc(nn: int, deg: list):
    nnn = np.repeat(nn, len(deg))
    return np.array(list(map(lambda ldeg, lnn: math.cos(math.radians(ldeg))**lnn, deg, nnn)))


def calMF(x0: list, data: list):

    global psc_final, fSN, fSE, fSV, fSNE, fSEV, fSVN

    psc = np.zeros(len(data[0]))

    [SN0, SE0, SV0, SNE0, SEV0, SVN0, alphaNN, alphaEE] = \
        [np.repeat(item, len(data[0])) for item in x0]
    den, tdepth, over = data[0:3]
    fractype, alpha, beta, dep, psm, psi, pi = data[3:10]

    depth = dep - tdepth

    alphaVV = den
    alphaNE = 0.5*(alphaNN-alphaEE)*2.*SNE0/(SN0-SE0)
    alphaEV = 0.5*(alphaEE-alphaVV)*2.*SEV0/(SE0-SV0)
    alphaVN = 0.5*(alphaVV-alphaNN)*2.*SVN0/(SV0-SN0)

    SN = SN0 + depth*alphaNN
    SE = SE0 + depth*alphaEE
    SV = SV0 + depth*alphaVV
    SNE = SNE0 + depth*alphaNE
    SEV = SEV0 + depth*alphaEV
    SVN = SVN0 + depth*alphaVN

    ver = tuple([fractype == 0])
    inc = tuple([fractype > 0])

    psc[inc] = SN[inc]*fc(2., pi)[inc]*fc(2., psi)[inc] + SE[inc]*fc(2., pi)[inc]*fs(2., psi)[inc] + SV[inc]*fs(2., pi)[inc] + SNE[inc]*fc(2., pi)[inc]*fs(1., 2.*psi)[inc] + SEV[inc]*fs(1., 2.*pi)[inc]*fs(1., psi)[inc] + SVN[inc]*fs(1., 2.*pi)[inc]*fc(1.,psi)[inc]

    psc_final = psc
    
    fSN, fSE, fSV, fSNE, fSEV, fSVN = [SN, SE, SV, SNE, SEV, SVN]

    errsum = (psm-psc)**2.
    errsum = (errsum.sum()/(len(errsum)-1))**.5

    return errsum


def BH_and_wsm_func(WSM_file, lat, lng):    
      
    wsm = pd.read_csv(WSM_file, encoding='ISO-8859-1', engine='python')
    kor = wsm[wsm['COUNTRY']=='Korea - Republic of']
    print(kor['LAT'])
    print(kor['LON'])
    kor['dist_wsm_src(km)'] = list(map(lambda slat,slng: haversine((lat,lng),(slat,slng), unit='km'), kor['LAT'], kor['LON']))
    kor_sorted = kor.sort_values(by=['dist_wsm_src(km)'])[['ID','dist_wsm_src(km)','TYPE','DEPTH','QUALITY','REGIME','LOCALITY','DATE','NUMBER','SD','METHOD','S1AZ','S1PL','S2AZ','S2PL','S3AZ','S3PL','MAG_INT_S1','SLOPES1','MAG_INT_S2','SLOPES2','MAG_INT_S3','SLOPES3']]
        
    m = folium.Map(location=[lat, lng], zoom_start=13)
    
    folium.Marker([lat,lng], popup=test,tooltip=test).add_to(m)
    kor.apply(lambda row:folium.CircleMarker([row['LAT'],row['LON']], popup=row['ID'], tooltip=row['ID'], radius=5, color='black', fill='gray').add_to(m), axis=1)
        
    st_folium(m, width=1800)
    st.markdown(f"### WSM data from the 5 points cloest to {test}")
    st.table(kor_sorted[0:5])
    st.markdown("World Stress Map https://datapub.gfz-potsdam.de/download/10.5880.WSM.2016.001/")


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


 
    
# Begin
st.set_page_config(
    page_title="HF3Dpy",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

WSM_file = '.\wsm2016.csv'

st.sidebar.title = "HF3Dpy"    
filename = file_selector()

if filename[-3:].lower()=="dat":
    st.sidebar.write('You selected `%s`' % filename)

    infofile = filename[:-3]+"info"

    if os.path.isfile(infofile):
        #try:
            f = open(infofile, mode='r', encoding='utf-8')
            info = f.readlines()
            test = info[1].replace('\n','')
            project = info[3].replace('\n','')
            date = pd.to_datetime(info[5].replace('\n',''))
            lat = pd.to_numeric(info[7].replace('\n','').split(",")[0])
            lng = pd.to_numeric(info[7].replace('\n','').split(",")[1])

            if test != '':
                st.markdown(f"## {test}")
            if project != '':    
                st.markdown(f"Project: {project}")
            if date:
                st.markdown(f"Date: {date:%Y-%m-%d}")
            if lat != 0:
                st.markdown(f"Latitude, longitude: {lat}, {lng}")
                BH_and_wsm_func(WSM_file,lat,lng)
        #except all:
            #st.sidebar.write("info file...")


    # reading dat file
    data_file = open(filename, mode='r')

    # line 1-3
    density, tdepth, tburden = [
        float(xx) for xx in data_file.readline().replace("\n", "").split('\t')[0:3]]
    x0 = [float(xx)
          for xx in data_file.readline().replace("\n", "").split('\t')]
    norows = int(data_file.readline().replace("\n", "").split('\t')[0])

    # line 4-
    temp = list()
    for ii in range(0, norows, 1):
        temp.append(data_file.readline().replace("\n", "").split('\t'))
    m = pd.DataFrame(temp, columns=[
        'findex', 'bbering', 'binclin', 'mdepth', 'psm', 'fstrike', 'fdip', 'dummy'])

    st.markdown("### Result")
    st.markdown(f"Rock density (10^6 kg/m3): {density}")
    st.markdown(f"Total depth from ground surface to upper boundary of bedrock (m): {tdepth}")
    st.markdown(f"Vertical stress of overburden (MPa): {tburden}")            

    # array
    cden = np.repeat(density, norows)
    cz0 = np.repeat(tdepth, norows)
    cburden = np.repeat(tburden, norows)
    fi = np.array(m.findex, dtype=int)
    bb = np.array(m.bbering, dtype=float)
    incl = np.array(m.binclin, dtype=float)
    mdep = np.array(m.mdepth, dtype=float)
    psm = np.array(m.psm, dtype=float)
    fstr = np.array(m.fstrike, dtype=float)
    fdip = np.array(m.fdip, dtype=float)

    data = [cden, cz0, cburden, fi, bb, incl, mdep, psm, fstr, fdip]

    result = minimize(calMF, x0, data, tol=1.e-6, method='BFGS')        

    df = pd.DataFrame([fi, mdep, cz0, mdep-cz0, psc_final, psm,
                      psm-psc_final, fSV-(cburden+cden*(mdep-cz0)), fSN, fSE, fSV, fSNE, fSEV, fSVN])
    df = df.T
    df.columns = ["Fracture_type", "mdepth", "tdepth", "depth", "Psc", "Psm",
                  "tolPs", "tolPv", "PN", "PE", "PV", "PNE", "PEV", "PVN"]        


    # ---- Stereonet -------
    mag = [[] for i in range(3)]
    vec = [[] for i in range(3)]
    for idx, [PN, PE, PV, PNE, PEV, PVN] in df[["PN", "PE", "PV", "PNE", "PEV", "PVN"]].iterrows():
        sigma = np.asarray([[PN, PNE, PVN],
                            [PNE, PE, PEV],
                            [PVN, PEV, PV]])

        e_val, e_vec = np.linalg.eig(sigma)
        # sort by magnitude of eigenvalues
        idx = e_val.argsort()[::-1]
        e_val = e_val[idx]
        e_vec = e_vec[:, idx]
        for ii in range(0, 3):
            mag[ii].append(np.round(e_val[ii],4))
            vec[ii].append(np.round(e_vec[ii],4))

    df["P1mag"] = mag[0]
    df["P2mag"] = mag[1]
    df["P3mag"] = mag[2]
    df["P1vec"] = vec[0]
    df["P2vec"] = vec[1]
    df["P3vec"] = vec[2]
    x1 = [xx[0] for xx in df["P1vec"]]
    y1 = [xx[1] for xx in df["P1vec"]]
    z1 = [xx[2] for xx in df["P1vec"]]
    x2 = [xx[0] for xx in df["P2vec"]]
    y2 = [xx[1] for xx in df["P2vec"]]
    z2 = [xx[2] for xx in df["P2vec"]]
    x3 = [xx[0] for xx in df["P3vec"]]
    y3 = [xx[1] for xx in df["P3vec"]]
    z3 = [xx[2] for xx in df["P3vec"]]            

    st.table(df)          


    ## fig 1
    gf1 = pd.concat([df.tolPs, df.mdepth], axis=1)
    gf1.rename(columns={"tolPs": "X"}, inplace=True)
    gf1["Legend"] = "Psm-Psc"
    gf2 = pd.concat([df.tolPv, df.mdepth], axis=1)
    gf2.rename(columns={"tolPv": "X"}, inplace=True)
    gf2["Legend"] = "PN-(tburden+depth*den)"
    gf = pd.concat([gf1, gf2])
    fig1 = px.scatter(
        gf,
        x="X",
        y="mdepth",
        color="Legend",
        labels=dict(
            X="Psm-Psc or PN-(tburden+depth*den) (MPa)",
            mdepth="Depth (m)",
            Legend="Stress",
        ),
        height=700,
    )
    fig1.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig1.update_yaxes(
        autorange="reversed",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )

    x_axis_max = max(max(abs(df.tolPs)), max(abs(df.tolPv)))

    fig1.update_xaxes(
        range=[-x_axis_max * 1.2, x_axis_max * 1.2],
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )

    fig1.update_layout(
        template="simple_white",
        font=dict(size=15,color="black"),
        legend=dict(font=dict(size=13)),
        margin=dict(l=100, r=150, t=50, b=50),
    )   


    ## fig 2
    gf1 = pd.concat([df.PN, df.mdepth], axis=1)
    gf1.rename(columns={"PN": "X"}, inplace=True)
    gf1["Legend"] = "PN"

    gf2 = pd.concat([df.PE, df.mdepth], axis=1)
    gf2.rename(columns={"PE": "X"}, inplace=True)
    gf2["Legend"] = "PE"

    gf3 = pd.concat([df.PV, df.mdepth], axis=1)
    gf3.rename(columns={"PV": "X"}, inplace=True)
    gf3["Legend"] = "PV"

    gfA = pd.concat([gf1, gf2, gf3])

    fig2 = px.scatter(
        gfA,
        x="X",
        y="mdepth",
        color="Legend",
        trendline="ols",
        labels=dict(X="PN or PE or PV (MPa)", mdepth="Depth (m)", Legend="Stress"),
        height=700,
    )

    PNEV_results = px.get_trendline_results(fig2).px_fit_results.iloc[0:3]

    fig2.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig2.update_yaxes(
        autorange="reversed",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )
    x_axis_max = max(
        max(df.PN), max(df.PE), max(df.PV), max(df.PNE), max(df.PEV), max(df.PVN)
    )
    x_axis_min = min(
        0,
        min(min(df.PN), min(df.PE), min(df.PV)),
        min(min(df.PNE), min(df.PEV), min(df.PVN)),
    )
    fig2.update_xaxes(
        range=[x_axis_min * 1.2, x_axis_max * 1.2],
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )

    fig2.update_layout(
        template="simple_white",
        font=dict(size=15,color="black"),
        legend=dict(font=dict(size=13)),
        margin=dict(l=100, r=150, t=50, b=50),
    )
    anno = f"<b>Y=a+bX</b><br><b>PN</b><br>a: {PNEV_results[0].params[0]:0.3f}<br>b: {PNEV_results[0].params[1]:0.3f}<br>r2: {PNEV_results[0].rsquared:0.3f}<br><b>PE</b><br>a: {PNEV_results[1].params[0]:0.3f}<br>b: {PNEV_results[1].params[1]:0.3f}<br>r2: {PNEV_results[1].rsquared:0.3f}<br><b>PV</b><br>a: {PNEV_results[2].params[0]:0.3f}<br>b: {PNEV_results[2].params[1]:0.3f}<br>r2: {PNEV_results[0].rsquared:0.3f}<br>"
    fig2.add_annotation(
        text=anno,
        font=dict(size=14),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.01,
        y=0.65,
        xanchor="left",
        yanchor="top",
        bordercolor="black",
        borderwidth=0,
    )



    gf4 = pd.concat([df.PNE, df.mdepth], axis=1)
    gf4.rename(columns={"PNE": "X"}, inplace=True)
    gf4["Legend"] = "PNE"

    gf5 = pd.concat([df.PEV, df.mdepth], axis=1)
    gf5.rename(columns={"PEV": "X"}, inplace=True)
    gf5["Legend"] = "PEV"

    gf6 = pd.concat([df.PVN, df.mdepth], axis=1)
    gf6.rename(columns={"PVN": "X"}, inplace=True)
    gf6["Legend"] = "PVN"

    gfB = pd.concat([gf4, gf5, gf6])

    fig4 = px.scatter(
        gfB,
        x="X",
        y="mdepth",
        color="Legend",
        trendline="ols",
        labels=dict(X="PNE or PEV or PVN (MPa)", mdepth="Depth (m)", Legend="Stress"),
        height=700,
    )

    PNEV_results = px.get_trendline_results(fig4).px_fit_results.iloc[0:3]

    fig4.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig4.update_yaxes(
        autorange="reversed",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )
    fig4.update_xaxes(
        range=[x_axis_min * 1.2, x_axis_max * 1.2],
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )

    fig4.update_layout(
        template="simple_white",
        font=dict(size=15,color="black"),
        legend=dict(font=dict(size=13)),
        margin=dict(l=100, r=150, t=50, b=50),
    )
    anno = f"<b>Y=a+bX</b><br><b>PNE</b><br>a: {PNEV_results[0].params[0]:0.3f}<br>b: {PNEV_results[0].params[1]:0.3f}<br>r2: {PNEV_results[0].rsquared:0.3f}<br><b>PEV</b><br>a: {PNEV_results[1].params[0]:0.3f}<br>b: {PNEV_results[1].params[1]:0.3f}<br>r2: {PNEV_results[1].rsquared:0.3f}<br><b>PVN</b><br>a: {PNEV_results[2].params[0]:0.3f}<br>b: {PNEV_results[2].params[1]:0.3f}<br>r2: {PNEV_results[2].rsquared:0.3f}<br>"
    fig4.add_annotation(
        text=anno,
        font=dict(size=14),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.01,
        y=0.65,
        xanchor="left",
        yanchor="top",
        bordercolor="black",
        borderwidth=0
    )
    

    row11, row12, row13 = st.columns(3)
    with row11:
        st.plotly_chart(fig1, use_container_width=True) 
    with row12:
        st.plotly_chart(fig2, use_container_width=True) 
    with row13:
        st.plotly_chart(fig4, use_container_width=True) 
        
        

# Set up the figure
    fig6 = plt.figure()
    
    ax1 = fig6.add_subplot(221, projection="stereonet")
    ax2 = fig6.add_subplot(223, projection="stereonet")
    ax3 = fig6.add_subplot(224, projection="stereonet")

    ax1.set_azimuth_ticks([])
    ax2.set_azimuth_ticks([])
    ax3.set_azimuth_ticks([])

    # Convert these to plunge/bearings for plotting.
    # Alternately, we could use xyz2stereonet (it doesn't correct for bi-directional
    # measurements, however) or vector2pole.
    plunge1, bearing1 = mplstereonet.vector2plunge_bearing(x1, y1, z1)
    plunge2, bearing2 = mplstereonet.vector2plunge_bearing(x2, y2, z2)
    plunge3, bearing3 = mplstereonet.vector2plunge_bearing(x3, y3, z3)
    strike1, dip1 = mplstereonet.vector2pole(x1, y1, z1)

    # Make a density contour plot of the orientations
    ax1.density_contourf(plunge1, bearing1, measurement="lines")
    ax1.line(plunge1, bearing1, marker="o", color="black")
    ax1.grid(True)
    ax1.set_title("Major Principal Stress", font=dict(size=9))
    #ax1.set_azimuth_ticks(range(0, 360, 10))

    ax2.density_contourf(plunge2, bearing2, measurement="lines")
    ax2.line(plunge2, bearing2, marker="o", color="black")
    ax2.grid(True)
    ax2.set_title("Intermediate Principal Stress", font=dict(size=9))
    #ax2.set_azimuth_ticks(range(0, 360, 10))

    ax3.density_contourf(plunge3, bearing3, measurement="lines")
    ax3.line(plunge3, bearing3, marker="o", color="black")
    ax3.grid(True)
    ax3.set_title("Minor Principal Stress", font=dict(size=9))
    #ax3.set_azimuth_ticks(range(0, 360, 10))
    
    
        
    
    
    
        # ---------
    gf1 = pd.concat([df.P1mag, df.mdepth], axis=1)
    gf1.rename(columns={"P1mag": "X"}, inplace=True)
    gf1["Legend"] = "P1"

    gf2 = pd.concat([df.P2mag, df.mdepth], axis=1)
    gf2.rename(columns={"P2mag": "X"}, inplace=True)
    gf2["Legend"] = "P2"

    gf3 = pd.concat([df.P3mag, df.mdepth], axis=1)
    gf3.rename(columns={"P3mag": "X"}, inplace=True)
    gf3["Legend"] = "P3"

    gfA = pd.concat([gf1, gf2, gf3])

    fig7 = px.scatter(
        gfA,
        x="X",
        y="mdepth",
        color="Legend",
        trendline="ols",
        labels=dict(
            X="Principal stress (MPa)", mdepth="Depth (m)", Legend="Principal<br>stress"
        ),
        height=700,
    )

    PRINCIPAL_results = px.get_trendline_results(fig7).px_fit_results.iloc[0:3]

    fig7.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig7.update_yaxes(
        autorange="reversed",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )
    
    fig7.update_xaxes(
        range=[x_axis_min * 1.2, x_axis_max * 1.2],
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )

    fig7.update_layout(
        template="simple_white",
        font=dict(size=15,color="black"),
        legend=dict(font=dict(size=13)),
        margin=dict(l=100, r=150, t=50, b=50),
    )
    
    anno = f"<b>Y=a+bX</b><br><b>P1</b><br>a: {PRINCIPAL_results[0].params[0]:0.3f}<br>b: {PRINCIPAL_results[0].params[1]:0.3f}<br>r2: {PRINCIPAL_results[0].rsquared:0.3f}<br><b>P2</b><br>a: {PRINCIPAL_results[1].params[0]:0.3f}<br>b: {PRINCIPAL_results[1].params[1]:0.3f}<br>r2: {PRINCIPAL_results[1].rsquared:0.3f}<br><b>P3</b><br>a: {PRINCIPAL_results[2].params[0]:0.3f}<br>b: {PRINCIPAL_results[2].params[1]:0.3f}<br>r2: {PRINCIPAL_results[2].rsquared:0.3f}<br>"
    
    fig7.add_annotation(
        text=anno,
        font=dict(size=14),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.01,
        y=0.65,
        xanchor="left",
        yanchor="top",
        bordercolor="black",
        borderwidth=0,
    )



    # ---------
    df["P12"] = df.P1mag / df.P2mag
    df["P13"] = df.P1mag / df.P3mag
    df["P23"] = df.P2mag / df.P3mag
    gf1 = pd.concat([df.P12, df.mdepth], axis=1)
    gf1.rename(columns={"P12": "X"}, inplace=True)
    gf1["Legend"] = "P1/P2"

    gf2 = pd.concat([df.P23, df.mdepth], axis=1)
    gf2.rename(columns={"P23": "X"}, inplace=True)
    gf2["Legend"] = "P2/P3"

    gf3 = pd.concat([df.P13, df.mdepth], axis=1)
    gf3.rename(columns={"P13": "X"}, inplace=True)
    gf3["Legend"] = "P1/P3"

    gfA = pd.concat([gf1, gf2, gf3])

    fig8 = px.scatter(
        gfA,
        x="X",
        y="mdepth",
        color="Legend",
        trendline="ols",
        labels=dict(X="Principal stress ratio", mdepth="Depth (m)", Legend="Legend"),
        height=700,
    )

    PRINCIPAL_results = px.get_trendline_results(fig8).px_fit_results.iloc[0:3]

    fig8.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig8.update_yaxes(
        autorange="reversed",
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        mirror=True,
    )
    
    fig8.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey", mirror=True)

    fig8.update_layout(
        template="simple_white",
        font=dict(size=15,color="black"),
        
        legend=dict(font=dict(size=13)),
        margin=dict(l=100, r=150, t=50, b=50),
    )
    
    anno = f"<b>Y=a+bX</b><br><b>P1/P2</b><br>a: {PRINCIPAL_results[0].params[0]:0.3f}<br>b: {PRINCIPAL_results[0].params[1]:0.3f}<br>r2: {PRINCIPAL_results[0].rsquared:0.3f}<br><b>P2/P3</b><br>a: {PRINCIPAL_results[1].params[0]:0.3f}<br>b: {PRINCIPAL_results[1].params[1]:0.3f}<br>r2: {PRINCIPAL_results[1].rsquared:0.3f}<br><b>P1/P3</b><br>a: {PRINCIPAL_results[2].params[0]:0.3f}<br>b: {PRINCIPAL_results[2].params[1]:0.3f}<br>r2: {PRINCIPAL_results[2].rsquared:0.3f}<br>"
    
    fig8.add_annotation(
        text=anno,
        font=dict(size=14),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=1.01,
        y=0.65,
        xanchor="left",
        yanchor="top",
        bordercolor="black",
        borderwidth=0,
    )






























    row21, row22, row23 = st.columns([1,1,1])
    with row21:
        st.plotly_chart(fig7, use_container_width=True) 
    with row22:
        st.plotly_chart(fig8, use_container_width=True) 
    with row23:
        st.pyplot(fig6, use_container_width=True)    
            




