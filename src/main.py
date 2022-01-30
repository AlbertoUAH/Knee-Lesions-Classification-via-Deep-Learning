# -- Knee MRI Final Application VIA Streamlit
# -- Author: Fernandez Hernandez, Alberto
# -- Date: 2022 - 01 - 28
# install streamlit library: !pip install streamlit

# -- Libraries
from   plotly         import subplots
from   utils          import *
import plotly.express as px
import streamlit      as st
import numpy          as np
import torch.nn       as nn
import torch
import glob
import cv2
import re

# -- Streamlit tools & setup
st.set_page_config(layout="wide")
st.write("Data source:")
st.image('https://stanfordmlgroup.github.io/img/stanfordmlgrouplogo.svg')

st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">MRI Knee Lesion Classification webapp</p>', unsafe_allow_html=True)

label_str = st.selectbox(
	'Select label/diagnosis',
	list(LABEL_DICT.keys())
)

files_str = st.selectbox(
	'Select patient id',
	set(map(lambda x: re.sub(r'(_[a-zA-Z]+)|(\.npy)', '', x.split('/')[-1]),
		glob.glob('./files/*')))
)

# -- Load model
model = torch.load('./models/mrnet_three_pretrained_models_non_frozen_weights_standarized_img_aug_gradcam_2022_01_28.pth',
				       map_location=torch.device("cpu"))

# -- Load images (.npy files)
files_str_selected = './data/' + files_str
img_axial   = prepare_data(np.load(files_str_selected + '_axial.npy'))
img_sagital = prepare_data(np.load(files_str_selected + '_sagittal.npy'))
img_coronal = prepare_data(np.load(files_str_selected + '_coronal.npy'))
label       = LABEL_DICT[label_str]

sample_img  = [img_axial, img_sagital, img_coronal]

# -- Build heatmaps
heatmap_axial, heatmap_sagittal, heatmap_coronal, proba = build_grad_cam(model, sample_img, label=label)

superimposed_img_axial, superimposed_img_sagittal, superimposed_img_coronal = superimpose_img_heatmap(img_axial, heatmap_axial),\
																			  superimpose_img_heatmap(img_sagital, heatmap_sagittal),\
																			  superimpose_img_heatmap(img_coronal, heatmap_coronal)

st.write("Diagnosis selected: {}".format(label_str))
st.write("Probability: {:0.2f} %".format(proba))
col1, col2, col3 = st.columns(3)

fig_axial = px.imshow(superimposed_img_axial, animation_frame=0, binary_string=False,
					  color_continuous_scale='RdBu_r', labels=dict(animation_frame="slice"))
fig_axial.update_xaxes(showticklabels=False)
fig_axial.update_yaxes(showticklabels=False)
fig_axial.layout.height = FIG_HEIGHT
fig_axial.layout.width  = FIG_WIDTH
fig_axial.update_traces(
   hoverinfo="none", hovertemplate=None
)
col1.plotly_chart(fig_axial, use_column_width=True)

fig_sagittal = px.imshow(superimposed_img_sagittal, animation_frame=0, binary_string=False,
						 color_continuous_scale='RdBu_r', labels=dict(animation_frame="slice"))
fig_sagittal.update_xaxes(showticklabels=False)
fig_sagittal.update_yaxes(showticklabels=False)
fig_sagittal.layout.height = FIG_HEIGHT
fig_sagittal.layout.width  = FIG_WIDTH
fig_sagittal.update_traces(
   hoverinfo="none", hovertemplate=None
)
col2.plotly_chart(fig_sagittal, use_column_width=True)

fig_coronal = px.imshow(superimposed_img_coronal, animation_frame=0, binary_string=False,
						color_continuous_scale='RdBu_r', labels=dict(animation_frame="slice"))
fig_coronal.update_xaxes(showticklabels=False)
fig_coronal.update_yaxes(showticklabels=False)
fig_coronal.layout.height = FIG_HEIGHT
fig_coronal.layout.width  = FIG_WIDTH
fig_coronal.update_traces(
   hoverinfo="none", hovertemplate=None
)
col3.plotly_chart(fig_coronal, use_column_width=True)