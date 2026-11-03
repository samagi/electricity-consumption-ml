import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title='Household electricity predictor')

@st.cache_resource
def load_bundle():
    return joblib.load('model.joblib')

b = load_bundle()
st.title('Household electricity consumption predictor')
st.write('Enter some household details. The model predicts the average monthly consumption (kWh) '
         'and shows which factors pushed the prediction up or down. All other details are set to typical values.')

inputs = {}
for f in b['ui_features']:
    inputs[f['name']] = st.slider(f['name'].replace('_', ' '), float(f['min']), float(f['max']),
                                  float(min(max(f['default'], f['min']), f['max'])), step=float(f['step']))

row = pd.DataFrame([b['defaults']])[b['columns']]
for k, v in inputs.items():
    row[k] = v
for c in b['cat_cols']:
    row[c] = pd.Categorical(row[c], categories=b['categories'][c])

pred = float(b['model'].predict(row)[0])
st.metric('Predicted consumption (kWh per month)', f'{pred:.0f}')

st.subheader('Why this prediction?')
sv = shap.TreeExplainer(b['model'])(row)
shap.plots.waterfall(sv[0], max_display=10, show=False)
st.pyplot(plt.gcf())
plt.close('all')
