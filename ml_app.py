import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from feature_engine.outliers import OutlierTrimmer
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
from category_encoders import TargetEncoder
import difflib
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

# Load ML Models

def load_df(df):
	df = pd.read_csv(df)
	return df

df = load_df("df_pop.csv")


def prediction(model, df, Commune, No_voie, type_voie, voie, lot, surface_bati, nbr_pieces, surface_terrain, type_bien, nature_mutation):
    Commune = Commune.upper()
    Commune = Commune.replace("-", " ")
    voie = voie.upper()
    type_bien = type_bien.upper()

    df_ville_avant = df[df.Commune == Commune]
    df_ville_avant["Population municipale"] = df_ville_avant["Population municipale"].fillna(df_ville_avant["Population municipale"].median())

    df_ville_avant["Code postal"] = df_ville_avant["Code postal"].astype("int").astype("object")
    df_ville_avant["Code departement"] = df_ville_avant["Code departement"].astype("object")

    capper = OutlierTrimmer(capping_method="iqr", tail='both', fold=1.5, variables="prix_m2", missing_values = "ignore")
    capper.fit(df_ville_avant)
    df_ville = capper.transform(df_ville_avant)

    num = df_ville[df_ville.dtypes[df_ville_avant.dtypes!= 'object'].index]
    cat = df_ville[df_ville.dtypes[(df_ville_avant.dtypes== 'object')|(df_ville_avant.dtypes=="category")].index]

    cat["prix_m2_cat"] = df_ville.prix_m2
    cat['No_voie_cat'] = df_ville["No voie"].astype(str)

    encoder_quartier = TargetEncoder()
    cat["quartier"] = cat['No_voie_cat'] + " " + cat.Voie
    cat['Quartier_enc'] = encoder_quartier.fit_transform(cat['quartier'], cat['prix_m2_cat'])

    le_type_voie = LabelEncoder()
    cat["Type de voie"] = le_type_voie.fit_transform(cat["Type de voie"])
    type_voie = le_type_voie.transform(np.array([type_voie]))


    le_Nature_mutation = LabelEncoder()
    cat["Nature mutation"] = le_Nature_mutation.fit_transform(cat["Nature mutation"])
    nature_mutation = le_Nature_mutation.transform(np.array([nature_mutation]))


    le_type_local = LabelEncoder()
    cat["Type local"] = le_type_local.fit_transform(cat["Type local"])
    type_bien = le_type_local.transform(np.array([type_bien]))

    df_ville = pd.DataFrame(pd.concat([cat, num], axis = 1),columns = [*cat.columns, *num.columns])

    df_ville = df_ville.drop(['prix_2020', 'Population municipale', "Nature culture",
                              'Code postal', "No voie", "quartier",
                             "cat_prix", "Region", "Commune", 'Code departement',
                              "Voie", 'No voie', 'Population municipale', "Commune", "prix_m2_cat", "No_voie_cat"], axis = 1)


    train = df_ville[df_ville.annee.isin([2016, 2017, 2018, 2019, 2020])]
    test = df_ville[df_ville.annee == 2021]

    X_train = train.drop("prix_m2", axis =1)
    y_train = train["prix_m2"]

    X_test = test.drop("prix_m2", axis =1)
    y_test = test["prix_m2"]

    #Standardisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    rmse_test = mean_squared_error(y_test, pred, squared=False)




    liste_quartier = cat["quartier"].unique()
    quartier = str(No_voie) + " " + voie

    best_match = difflib.get_close_matches(quartier, liste_quartier, n=1, cutoff =0.5)
    quartier_enc = cat[cat["quartier"]==best_match[0]].prix_m2_cat.mean()

    liste_voies = df_ville_avant.Voie.unique()
    voie =  difflib.get_close_matches(voie, liste_voies, n=1, cutoff =0.1)

    latitude = df_ville_avant[df_ville_avant.Voie == voie[0]].latitude.mean()
    longitude = df_ville_avant[df_ville_avant.Voie == voie[0]].longitude.mean()

    mois = datetime.now().month
    annee = datetime.now().year

    X = np.array([[nature_mutation, type_voie, type_bien, quartier_enc,
       lot, surface_bati, nbr_pieces, surface_terrain, mois, annee,
       latitude, longitude]])
    X_float = X.astype(float)
    X_scaled = scaler.transform(X_float)

    resultat = model.predict(X_scaled)

    prix = resultat*surface_bati
    prix = prix[0]
    prix_min = prix - surface_bati*rmse_test
    prix_max = prix + surface_bati*rmse_test

    return f"Selon l'état de votre bien le prix estimé est entre {round(prix_min)}€ et {round(prix_max)}€"







def run_ml_app():
	st.subheader("Machine Learning Section")
	st.write("Rentrez les caractéristiques de votre bien immobilier pour évaluer son prix de vente")
	page = st.sidebar.selectbox("Modèles",["XGBoostRegressor","RandomForestRegressor", "CatBoostRegressor"])

	if page == "XGBoostRegressor":
		model = XGBRegressor()
	elif page == "RandomForestRegressor":
		model = RandomForestRegressor()
	elif page == "CatBoostRegressor":
		model = CatBoostRegressor()



	col1, col2 = st.columns(2)
	with col1:
		type_bien = st.radio("Type de bien", ("Maison", "Appartement"))

		nbr_pieces = st.number_input("Entrez le nombre de pièces principales", min_value=0, max_value=None)
		Commune = st.text_input('Entrer la ville')
		No_voie = st.number_input("Entrez le No de voie", min_value=1, max_value=None)
		voie = st.text_input('Entrer la voie')
		voie = voie.upper()


	with col2:
		nature_mutation = st.radio("Nature de mutation", ("Vente", "Terrain à bâtir"))
		surface_bati = st.number_input("Entrez la surface du bien", min_value=0, max_value=None)
		surface_terrain = st.number_input("Entrez la surface du terrain", min_value=0, max_value=None)
		type_voie = st.selectbox("Type de voie", ("Rue", "Route", "Allée", "Route", "Chemin", "Impasse",
									   "Place", "Boulevard", "Résidennce", "Autre"))
		type_voie=type_voie.upper()
		lot = st.number_input("Nombre de lots", min_value=0, max_value=None)


	ok = st.button("Estimer la valeur de votre bien")
	if ok:

		resultat = prediction(model, df, Commune, No_voie, type_voie, voie, lot, surface_bati, nbr_pieces, surface_terrain, type_bien, nature_mutation)
		st.subheader(resultat)
		if type_bien == "Maison":
			maison = Image.open('maison.jpg')
			st.image(maison)
		elif type_bien == "Appartement":
			appartement = Image.open('appartement.jpg')
			st.image(appartement)















# # 	# Layout
# # 	col1,col2 = st.beta_columns(2)

# # 	with col1:
# # 		age = st.number_input("Age",10,100)
# # 		gender = st.radio("Gender",("Female","Male"))
# # 		polyuria = st.radio("Polyuria",["No","Yes"])
# # 		polydipsia = st.radio("Polydipsia",["No","Yes"])
# # 		sudden_weight_loss = st.selectbox("Sudden_weight_loss",["No","Yes"])
# # 		weakness = st.radio("weakness",["No","Yes"])
# # 		polyphagia = st.radio("polyphagia",["No","Yes"])
# # 		genital_thrush = st.selectbox("Genital_thrush",["No","Yes"])
# #
# #
# # 	with col2:
# # 		visual_blurring = st.selectbox("Visual_blurring",["No","Yes"])
# # 		itching = st.radio("itching",["No","Yes"])
# # 		irritability = st.radio("irritability",["No","Yes"])
# # 		delayed_healing = st.radio("delayed_healing",["No","Yes"])
# # 		partial_paresis = st.selectbox("Partial_paresis",["No","Yes"])
# # 		muscle_stiffness = st.radio("muscle_stiffness",["No","Yes"])
# # 		alopecia = st.radio("alopecia",["No","Yes"])
# # 		obesity = st.select_slider("obesity",["No","Yes"])

# # 	with st.beta_expander("Your Selected Options"):
# # 		result = {'age':age,
# # 		'gender':gender,
# # 		'polyuria':polyuria,
# # 		'polydipsia':polydipsia,
# # 		'sudden_weight_loss':sudden_weight_loss,
# # 		'weakness':weakness,
# # 		'polyphagia':polyphagia,
# # 		'genital_thrush':genital_thrush,
# # 		'visual_blurring':visual_blurring,
# # 		'itching':itching,
# # 		'irritability':irritability,
# # 		'delayed_healing':delayed_healing,
# # 		'partial_paresis':partial_paresis,
# # 		'muscle_stiffness':muscle_stiffness,
# # 		'alopecia':alopecia,
# # 		'obesity':obesity}
# # 		st.write(result)
# # 		encoded_result = []
# # 		for i in result.values():
# # 			if type(i) == int:
# # 				encoded_result.append(i)
# # 			elif i in ["Female","Male"]:
# # 				res = get_value(i,gender_map)
# # 				encoded_result.append(res)
# # 			else:
# # 				encoded_result.append(get_fvalue(i))


# # 		# st.write(encoded_result)
# # 	with st.beta_expander("Prediction Results"):
# # 		single_sample = np.array(encoded_result).reshape(1,-1)

# #
# # 		prediction = loaded_model.predict(single_sample)
# # 		pred_prob = loaded_model.predict_proba(single_sample)
# # 		st.write(prediction)
# # 		if prediction == 1:
# # 			st.warning("Positive Risk-{}".format(prediction[0]))
# # 			pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
# # 			st.subheader("Prediction Probability Score")
# # 			st.json(pred_probability_score)
# # 		else:
# # 			st.success("Negative Risk-{}".format(prediction[0]))
# # 			pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
# # 			st.subheader("Prediction Probability Score")
# # 			st.json(pred_probability_score)

