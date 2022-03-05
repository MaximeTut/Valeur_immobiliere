import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
from feature_engine.outliers import OutlierTrimmer
from streamlit_folium import folium_static
import folium
from IPython.display import display
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
sns.set()

def load_data(data):
	df = pd.read_csv(data)
	return df



def investir_ville(df, sup = 10**9, inf = 0, region = None, type_bien = None, exception = None):
    #On défini la population minimum en paramètre

    df = df[df["Population municipale"]<sup]
    df = df[df["Population municipale"]>inf]

    if region != None and region != "Toute la France" and region !="Tout sauf l'île de France" :
        region = region.upper()
        df = df[df["Region"] == region]
    if region == "Toute la France":
        df=df

    if region == "Tout sauf l'île de France":
        df = df[df["Region"]!="ILE DE FRANCE"]


    if type_bien == "Les deux":
        df =df
    else :
        df=df[df["Type local"]==type_bien]



    x_communes = []
    communes = []
    for commune in df.Commune.unique():
    #Je fais un group by pour chaque commune par année et par mois avec un reset_index pour obtenir un dataset par commune
        try :
            df_commune = df[df.Commune == commune]
            capper = OutlierTrimmer(capping_method="iqr", tail='both', fold=1.5, variables="prix_m2", missing_values = "ignore")
            capper.fit(df_commune)
            df_commune = capper.transform(df_commune)

            df_commune = df_commune.groupby(by = ["annee", "mois"]).agg({"prix_m2":"mean"}).reset_index()
            #Je prends la moyenne glissante sur trois mois
            df_commune["roll"]= df_commune.prix_m2.rolling(3).mean()

            #Je calcule l'équation de la regression linéaire simple avec polyfit de numpy
            z = np.polyfit(df_commune.index[2:], df_commune.roll[2:], deg = 1)
            x = z[0]

            #Je mets tous les x des communes dans une liste
            x_communes.append(x)
            communes.append(commune)

        except :
            continue

    x_df = pd.DataFrame({"coefficients": x_communes, "Commune": communes})

    x_df = x_df.sort_values("coefficients", ascending=False)
    fig = plt.figure()

    #Je prends les 10  plus hauts coefficients et les 10 plus faibles

    sns.heatmap(x_df[["coefficients", "Commune"]].set_index("Commune").head(10), annot = True, cmap ="Greens",
                linewidths = 2)
    plt.title("Investir", fontsize = 15)
    plt.ylabel('')
    st.pyplot(fig)

    fig = plt.figure()
    sns.heatmap(x_df[["coefficients", "Commune"]].set_index("Commune").tail(10), annot = True, cmap ="Reds_r",
                linewidths = 2)
    plt.title("Ne pas Investir", fontsize = 15)
    plt.ylabel('')
    st.pyplot(fig)

def afficher_tendance(df, ville):
	ville = ville.upper()
	df_commune = df[df.Commune == ville]
	df_commune_heat = df[df.Commune == ville]
	capper = OutlierTrimmer(capping_method="iqr", tail='both', fold=1.5, variables="prix_m2", missing_values = "ignore")
	capper.fit(df_commune)
	df_commune = capper.transform(df_commune)

    #On va prendre une moyenne glissante pour avoir une tendance plus claire
	df_commune_groupby = df_commune.groupby(by = ["annee", "mois"]).agg({"prix_m2":"median"}).reset_index()
	df_commune_groupby["roll"]= df_commune_groupby.prix_m2.rolling(3).mean()
	df_commune_groupby.mois = df_commune_groupby.mois.replace({1:"jan", 2:"fev", 3:"mar", 4:'avr', 5:"mai", 6:"juin", 7:"juil", 8:'aou', 9:"sep", 10:"oct", 11:"nov", 12:"dec"})
	df_commune_groupby["date"] = df_commune_groupby.mois.astype(str)+" "+df_commune_groupby.annee.astype(str)
	df_commune_groupby = df_commune_groupby.drop(["annee", "mois"], axis =1)

    #On calcule l'équation de la régression linéaire
	z = np.polyfit(df_commune_groupby.index[2:], df_commune_groupby.roll[2:], deg = 1)
	x = z[0]
	b = z[1]
	fig = plt.figure(figsize=(20,8))
	g = sns.lineplot(df_commune_groupby.date, df_commune_groupby.roll)
	sns.lineplot(df_commune_groupby.date, y = df_commune_groupby.index*x+b)
	plt.title(f"Courbe de tendance du prix au m2 de la ville de {ville}", fontsize = 35)
	g.set_xticklabels(df_commune_groupby.date, rotation=90)
	st.pyplot(fig)


	text_x= 30
	text_y= int(df_commune_groupby["roll"].min())
	plt.text(text_x, text_y+200, f"Le prix de la ville de {ville} augmente de {round(z[0])} euros par mois en moyenne", fontsize = 15)

    #On affiche la distribution des prix de la ville
	fig = plt.figure(figsize=(20,8))
	sns.distplot(df_commune[(df_commune.Commune == ville) &((df_commune.annee == 2021) | (df_commune.annee == 2020))].prix_m2, bins = 100, color = 'red')
	plt.title(f'Distribution des prix de la ville de {ville}', fontsize = 35)
	st.pyplot(fig)
    #J'affiche également le nombre de ventes afin d'évaluer la solidité des mesures
	fig = plt.figure(figsize=(20,8))
	sns.heatmap(pd.crosstab(df_commune_heat.annee, df_commune_heat.mois), annot = True, fmt="g", cmap = "RdBu_r");
	plt.title(f'Nombre de ventes de la ville de {ville} par mois', fontsize = 35)
	st.pyplot(fig)

def afficher_carte(df, ville, nbr_points):

    ville = ville.upper()
    df_ville = df[df.Commune == ville]
    df_ville["No voie"] = df_ville["No voie"].astype(int).astype(str)
    df_ville["Code postal"] = df_ville["Code postal"].astype(int).astype(str)
    df_ville["Type de voie"] = df_ville["Type de voie"].replace({"RTE": "route",
                                                          "CHE": "chemin",
                                                          "BD": "boulevard",
                                                          "PL":"place",
                                                          "AV":"avenue",
                                                          "IMP":"impasse",
                                                          "autre":"rue",
                                                          "ALL":"allée"})
    capper = OutlierTrimmer(capping_method='iqr', tail='both', fold=1.5, variables="prix_m2", missing_values = "ignore")
    capper.fit(df_ville)
    df_ville = capper.transform(df_ville)

    if nbr_points < df_ville.shape[0] :
        df_ville = df_ville.sample(n=nbr_points)
    else :
        df_ville = df_ville

    df_ville["adresse"] = df_ville["No voie"]+ " " + df_ville["Type de voie"] + " " + df_ville["Voie"] +" "+  df_ville["Code postal"] +" "+df_ville["Commune"]
    df_ville.dropna()

    #Je trouve les coordonnées avec l'adresse
    geolocator = Nominatim(user_agent="Maxime")
    lats_adresse = []
    long_adresse = []

    for index, row in df_ville.iterrows():
        try :
            adresse = row["adresse"]
            lat = geolocator.geocode(adresse)[1][0]
            long = geolocator.geocode(adresse)[1][1]

            lats_adresse.append(lat)
            long_adresse.append(long)
        except :
            lats_adresse.append("missing")
            long_adresse.append("missing")
            continue

    df_ville["lat"] = lats_adresse
    df_ville["long"] = long_adresse
    df_ville = df_ville[~df_ville.eq("missing").any(1)]


    lat = geolocator.geocode(ville)[1][0]
    long = geolocator.geocode(ville)[1][1]

    location = [lat, long]
    ville_map = folium.Map(location, zoom_start = 13)
    df_ville['marker_color'] = pd.cut(df_ville['prix_m2'], bins=3,
                              labels=['green', 'orange', 'red'])

    for row in df_ville.iterrows():
        row_values = row[1]
        location = [row_values['lat'], row_values['long']]

        tooltip = f'Prix :{round(row_values["prix_m2"])}€/m2<br>\
                Surface : {round(row_values["Surface reelle bati"])} m2<br>\
                Prix de vente : {round(row_values["prix_m2"])*round(row_values["Surface reelle bati"])}€<br>\
                Année de vente : {row_values["annee"]}<br>\
                Type de bien : {row_values["Type local"]}'

        marker = folium.Marker(location,radius=15, tooltip=tooltip, icon=folium.Icon(icon = "home",color=row_values["marker_color"]))
        marker.add_to(ville_map)

    folium_static(ville_map)






