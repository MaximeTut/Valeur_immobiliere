import streamlit as st
import streamlit.components.v1 as stc
from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
		<div style="background-color:#7E5109;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Etude des ventes foncières en France </h1>
		<h4 style="color:white;text-align:center;">De 2016 à 2021 </h4>
		</div>
		"""

def main():
	# st.title("ML Web App with Streamlit")
	stc.html(html_temp)

	menu = ["Home","Exploration","Machine learning"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.write("""

			### Investir dans l'immobilier en France
			#### Datasource
				- https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/
			""")
	elif choice == "Exploration":
		run_eda_app()
	elif choice == "Machine learning":
		run_ml_app()


if __name__ == '__main__':
	main()