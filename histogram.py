from utils.tools.UTIL import *
from scipy import stats
from plotly.subplots import make_subplots

import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter
import colorsys
import base64
from io import BytesIO



def adjust_color_lightness(color, amount=0.5):
    try:
        c = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        c = colorsys.rgb_to_hls(*[x/255.0 for x in c])
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount*c[1])), c[2])
    except ValueError:
        return (1, 1, 1)

def generate_color_scale(base_color):
    return [
        [int(255*r), int(255*g), int(255*b)]
        for r, g, b in [adjust_color_lightness(base_color, amount) for amount in np.linspace(0.3, 1.3, 10)]
    ]

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def create_histogram(datasets, title="Histogram", x_label="Value", y_label="Frequency",
                    bin_size=None, colors=None, show_text=True, cumulative=False, 
                    bargap=0.1, bargroupgap=0.2, NormGraph=True, NormGraphstyle='선',
                    show_control_limits=False):
    fig = make_subplots()
    
    # Add grid first
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='lightgray', width=1, dash='dot'),
        showlegend=False,
        xaxis='x',
        yaxis='y'
    ))
    
    def add_histogram(data, name, color):
        if bin_size:
            xbins = dict(start=min(data), end=max(data), size=bin_size)
        else:
            xbins = None
        
        hist = go.Histogram(
            x=data,
            name=name,
            marker_color=color, #f'rgb({color[0]},{color[1]},{color[2]})',
            opacity=1.0,
            cumulative_enabled=cumulative,
            xbins=xbins,
            texttemplate="%{y}" if show_text else None,
            textposition="outside",
            textfont=dict(size=15, color="black", family="Arial, sans-serif", weight="bold"),
        )
        
        fig.add_trace(hist)
        
        if NormGraph or show_control_limits:
            # Calculate the parameters for the normal distribution
            mu, std = np.mean(data), np.std(data)
            x = np.linspace(min(data), max(data), 100)
            p = stats.norm.pdf(x, mu, std)

            # Scale the normal distribution to match the histogram height
            hist_max = max(np.histogram(data)[0])
            p_scaled = p * (hist_max / max(p))

        if NormGraph:
            if NormGraphstyle == '선':
                linestyle = 'solid'
            else:
                linestyle = 'dot'
            # Add the normal distribution curve
            fig.add_trace(go.Scatter(
                x=x,
                y=p_scaled,
                mode='lines',
                line=dict(color=f'rgb({color[0]+20},{color[1]+20},{color[2]+20})', width=2, dash=linestyle),
                name=f'{name} Norm'
            ))
        
    
    
    for i, (name, data) in enumerate(datasets.items()):
        add_histogram(data, name, st.session_state[f'color_{i}'])
        
    if show_control_limits:
        # Add UCL and LCL
        ucl = st.session_state.UCL+.15
        lcl = st.session_state.LCL-.15
        fig.add_shape(type="line", x0=ucl, x1=ucl, y0=0, y1=max(np.histogram(data)[0]),
                    line=dict(color="red", width=2, dash="dash"),
                    name=f"UCL ({name})")
        fig.add_shape(type="line", x0=lcl, x1=lcl, y0=0, y1=max(np.histogram(data)[0]),
                    line=dict(color="red", width=2, dash="dash"),
                    name=f"LCL ({name})")
        # Add annotations for UCL and LCL
        fig.add_annotation(x=ucl, y=max(np.histogram(data)[0]), text="UCL", showarrow=True, arrowhead=2)
        fig.add_annotation(x=lcl, y=max(np.histogram(data)[0]), text="LCL", showarrow=True, arrowhead=2)
    
    barmode = 'group' if len([d for d in datasets.values() if d]) > 1 else 'relative'
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        bargap=bargap,
        bargroupgap=bargroupgap,
        barmode=barmode,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showline=False, showgrid=False, zeroline=False)
    fig.update_yaxes(
        showline=False, 
        showgrid=True, 
        gridcolor='lightgray', 
        gridwidth=1, 
        griddash='dot',
        zeroline=False,
        range=[0, max([max(np.histogram(d)[0]) for d in datasets.values()])+1]
    )
    
    return fig

def get_image_download_link(fig, filename="histogram.png", text="Download PNG"):
    buf = BytesIO()
    fig.write_image(buf, format="png", scale=2, width=1000, height=600)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def descriptive_statistics_set():
    st.subheader("Interactive Histogram")
    df = convert_to_calculatable_df()
    columns = df.columns.tolist()
        
    with st.expander("Select Data Columns", expanded=True):
        num_datasets = st.multiselect("Select columns", options=columns[:10], default=df.columns.tolist()[0])
        values = [df[col].dropna().values.tolist() for col in num_datasets]
        
        st.session_state.datasets = {}
        for i in range(len(num_datasets)):
            st.session_state.datasets[f"{columns[i]}"] = values[i]

    with st.expander("Color Palette", expanded=True):
        color_intensity = st.slider("Color Intensity", 1, 10, 5)
        BASE_COLORS = [
                    "#A2D2FF", "#A8E6CF", "#8EEDF7", "#FF9AA2", "#FDFD96",
                    "#FFB347", "#C7A0D9", "#FF9CE3", "#B0C4DE", "#A6D1D4"]

        COLOR_SCALES = [generate_color_scale(color) for color in BASE_COLORS]
        st.session_state.colors = [COLOR_SCALES[i % len(COLOR_SCALES)][color_intensity-1] for i in range(len(num_datasets))]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col6, col7, col8, col9, col10 = st.columns(5)
    
        for i in range(len(num_datasets)):
            if i == 0:
                col1.color_picker(f"{columns[i]}", 
                                rgb_to_hex(st.session_state.colors[i]), 
                                key=f"color_{i}")
            elif i == 1:
                col2.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 2:
                col3.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 3:
                col4.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 4:
                col5.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 5:
                col6.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 6:
                col7.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 7:
                col8.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 8:
                col9.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
            elif i == 9:
                col10.color_picker(f"{columns[i]}", rgb_to_hex(st.session_state.colors[i]), key=f"color_{i}")
    
    with st.expander("Edit Histogram Options", expanded=False):
        st.session_state.title = st.text_input("Title", "My Histogram")
        st.session_state.x_label = st.text_input("X-axis Label", "Value")
        st.session_state.y_label = st.text_input("Y-axis Label", "Frequency")
        
        st.session_state.bin_size = st.number_input("Bin Size (leave 0 for auto)", min_value=0.0, value=0.0, step=0.5)
        st.session_state.bin_size = st.session_state.bin_size if st.session_state.bin_size > 0 else None
        
        st.session_state.show_text = st.checkbox("Show frequency values", value=True)
        st.session_state.cumulative = st.checkbox("누적 히스토그램", value=False)
        
        st.session_state.bargap = st.slider("Bar Gap", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        st.session_state.bargroupgap = st.slider("Bar Group Gap", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    with st.expander("ETC Options", expanded=False):
        st.session_state.NormGraph = st.checkbox("정규선 그림", value=False)
        if st.session_state.NormGraph:
            st.selectbox('정규선 형태', ['선', '점'], key='NormGraphstyle')
            
        st.session_state.show_control_limits = st.checkbox("Control Limits 표시", value=False)
        if st.session_state.show_control_limits:
            st.number_input("UCL Limits", min_value=0.0, value=0.0, step=0.5, key='UCL')
            st.number_input("LCL Limits", min_value=0.0, value=0.0, step=0.5, key='LCL')

def descriptive_statistics_cal(df):
    datasets = st.session_state.datasets
    title = st.session_state.title
    x_label = st.session_state.x_label
    y_label = st.session_state.y_label
    bin_size = st.session_state.bin_size
    colors = st.session_state.colors
    show_text = st.session_state.show_text
    cumulative = st.session_state.cumulative
    bargap = st.session_state.bargap
    bargroupgap = st.session_state.bargroupgap
    NormGraph = st.session_state.NormGraph
    
    NormGraphstyle = '선'
    if st.session_state.NormGraph:
        NormGraphstyle = st.session_state.NormGraphstyle
        
    show_control_limits = False
    if st.session_state.show_control_limits:
        show_control_limits = st.session_state.show_control_limits
    
    fig = create_histogram(datasets, title, x_label, y_label,
                                bin_size, colors, show_text, cumulative,
                                bargap, bargroupgap, NormGraph, NormGraphstyle,
                                show_control_limits)
        
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
        
def descriptive_statistics_plot(df):
    pass

def descriptive_statistics_run():
    # 매우 중요 : 데이터프레임 가져오기 ------------
    df = convert_to_calculatable_df()
    # --------------------------------------
    descriptive_statistics_cal(df)
    descriptive_statistics_plot(df)