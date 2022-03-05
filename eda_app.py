import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
from feature_engine.outliers import OutlierTrimmer
#import geopandas as gpd
import folium
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
sns.set()
from eda_page import investir_ville
from eda_page import afficher_tendance
from eda_page import afficher_carte

regions = ('AUVERGNE RHÔNE ALPES', 'BOURGOGNE FRANCHE COMTE',
       'OCCITANIE', 'NORMANDIE', 'NOUVELLE AQUITAINE', 'GRAND EST',
       'HAUTS DE FRANCE', 'PAYS DE LA LOIRE', 'CENTRE VAL DE LOIRE',
       'BRETAGNE', 'ILE DE FRANCE', "PROVENCE ALPES CÔTE D'AZUR",
	   "Toute la France", "Tout sauf l'île de France")

@st.cache
def load_data(data):
	df = pd.read_csv(data)
	return df

df_pop = load_data("df_pop.csv")

def run_eda_app():
	st.subheader("Exploration")


	page = st.sidebar.selectbox("Menu exploration",["Investir","Tendance", "Carte"])
	if page == "Investir":
		st.write("Choisissez la taille de ville dans lesquelles vous souhaitez investir en précisantla population max et la population min")
		sup = st.number_input('Population max', min_value=5000, max_value=1000000, step=10000)
		inf = st.number_input('Population min', min_value=5000, max_value=1000000, step=10000)
		region = st.selectbox("Région", regions)
		type_bien = st.radio("Choisir le type de bien", ("MAISON", "APPARTEMENT", "Les deux"))


		ok = st.button("Où investir ?")
		if ok:
			investir_ville(df_pop, sup = sup, inf = inf, region = region, type_bien = type_bien, exception = None)

	if page == "Tendance":
		ville = st.text_input('Entrez une ville')
		ok = st.button("Afficher les tendances de la ville")
		if ok:
			afficher_tendance(df_pop, ville)

	if page == "Carte":
		nbr_points = st.slider("Nombre de points à afficher",5,300,10)
		ville = st.text_input('Entrez une ville')
		ok =st.button("Afficher la carte de la ville")
		if ok:
			afficher_carte(df = df_pop, ville = ville, nbr_points=nbr_points)









# 		# Layouts
# 		col1,col2 = st.beta_columns([2,1])
# 		with col1:
# 			with st.beta_expander("Dist Plot of Gender"):
# 				# fig = plt.figure()
# 				# sns.countplot(df['Gender'])
# 				# st.pyplot(fig)

# 				gen_df = df['Gender'].value_counts().to_frame()
# 				gen_df = gen_df.reset_index()
# 				gen_df.columns = ['Gender Type','Counts']
# 				# st.dataframe(gen_df)
# 				p01 = px.pie(gen_df,names='Gender Type',values='Counts')
# 				st.plotly_chart(p01,use_container_width=True)

# 			with st.beta_expander("Dist Plot of Class"):
# 				fig = plt.figure()
# 				sns.countplot(df['class'])
# 				st.pyplot(fig)





# 		with col2:
# 			with st.beta_expander("Gender Distribution"):
# 				st.dataframe(df['Gender'].value_counts())

# 			with st.beta_expander("Class Distribution"):
# 				st.dataframe(df['class'].value_counts())
#

# 		with st.beta_expander("Frequency Dist Plot of Age"):
# 			# fig,ax = plt.subplots()
# 			# ax.bar(freq_df['Age'],freq_df['count'])
# 			# plt.ylabel('Counts')
# 			# plt.title('Frequency Count of Age')
# 			# plt.xticks(rotation=45)
# 			# st.pyplot(fig)

# 			p = px.bar(freq_df,x='Age',y='count')
# 			st.plotly_chart(p)

# 			p2 = px.line(freq_df,x='Age',y='count')
# 			st.plotly_chart(p2)

# 		with st.beta_expander("Outlier Detection Plot"):
# 			# outlier_df =
# 			fig = plt.figure()
# 			sns.boxplot(df['Age'])
# 			st.pyplot(fig)

# 			p3 = px.box(df,x='Age',color='Gender')
# 			st.plotly_chart(p3)

# 		with st.beta_expander("Correlation Plot"):
# 			corr_matrix = df_clean.corr()
# 			fig = plt.figure(figsize=(20,10))
# 			sns.heatmap(corr_matrix,annot=True)
# 			st.pyplot(fig)

# 			p3 = px.imshow(corr_matrix)
# 			st.plotly_chart(p3)










