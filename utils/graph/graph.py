import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import colorsys
import base64
from io import BytesIO

BASE_COLORS = [
    "#A2D2FF", "#8EEDF7", "#A8E6CF", "#FF9AA2", "#FDFD96",
    "#FFB347", "#C7A0D9", "#FF9CE3", "#B0C4DE", "#A6D1D4"
]

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

COLOR_SCALES = [generate_color_scale(color) for color in BASE_COLORS]

def create_histogram(datasets, title="Histogram", x_label="Value", y_label="Frequency",
                    bin_size=None, colors=None, show_text=True, cumulative=False, 
                    bargap=0.1, bargroupgap=0.2):
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
            marker_color=f'rgb({color[0]},{color[1]},{color[2]})',
            opacity=0.8,
            cumulative_enabled=cumulative,
            xbins=xbins,
            texttemplate="%{y}" if show_text else None,
            textposition="outside",
            textfont_size=10,
        )
        
        fig.add_trace(hist)
    
    for i, (name, data) in enumerate(datasets.items()):
        if data:
            add_histogram(data, name, colors[i])
    
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
        zeroline=False
    )
    
    return fig

def get_image_download_link(fig, filename="histogram.png", text="Download PNG"):
    buf = BytesIO()
    fig.write_image(buf, format="png", scale=2, width=1000, height=600)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    st.title("Interactive Histogram Generator")
    
    num_datasets = st.sidebar.number_input("Number of datasets", min_value=1, max_value=10, value=1)
    color_intensity = st.sidebar.slider("Global Color Intensity", 1, 10, 5)
    
    datasets = {}
    colors = [COLOR_SCALES[i % len(COLOR_SCALES)][color_intensity-1] for i in range(num_datasets)]
    
    with st.sidebar.expander("Data & Color Palette", expanded=True):
        for i in range(num_datasets):
            col1, col2 = st.columns([3, 1])
            with col1:
                data_input = st.text_input(f"Data {i+1}", 
                                        value="1,2,3,4,5,1,2,3,2,1" if i == 0 else "",
                                        key=f"data_input_{i}")
            with col2:
                st.color_picker(f"Color {i+1}", rgb_to_hex(colors[i]), key=f"color_{i}", disabled=True)
            
            if data_input:
                datasets[f"Data {i+1}"] = [float(x) for x in data_input.split(',') if x.strip()]
    
    st.sidebar.header("Histogram Options")
    with st.sidebar.expander("Edit Histogram Options", expanded=True):
        title = st.text_input("Title", "My Histogram")
        x_label = st.text_input("X-axis Label", "Value")
        y_label = st.text_input("Y-axis Label", "Frequency")
        
        bin_size = st.number_input("Bin Size (leave 0 for auto)", min_value=0.0, value=0.0, step=0.5)
        bin_size = bin_size if bin_size > 0 else None
        
        show_text = st.checkbox("Show frequency values", value=True)
        cumulative = st.checkbox("Cumulative histogram", value=False)
        
        bargap = st.slider("Bar Gap", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        bargroupgap = st.slider("Bar Group Gap", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    if datasets:
        fig = create_histogram(datasets, title, x_label, y_label,
                                bin_size, colors, show_text, cumulative,
                                bargap, bargroupgap)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
        
        with st.popover("View Data Statistics"):
            for name, data in datasets.items():
                if data:
                    st.subheader(name)
                    st.write(f"Mean: {np.mean(data):.2f}")
                    st.write(f"Median: {np.median(data):.2f}")
                    st.write(f"Standard Deviation: {np.std(data):.2f}")
                    st.write(f"Mode: {Counter(data).most_common(1)[0][0]}")
    else:
        st.write("Please enter data for at least one dataset.")

if __name__ == "__main__":
    main()