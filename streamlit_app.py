import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import traceback # Importer le module traceback

# ... (le reste de votre code) ...

if submitted:
    client_data_for_api = {
        "EXT_SOURCE_1": EXT_SOURCE_1, "EXT_SOURCE_3": EXT_SOURCE_3, "AMT_CREDIT": AMT_CREDIT,
        "DAYS_BIRTH": DAYS_BIRTH, "EXT_SOURCE_2": EXT_SOURCE_2, "AMT_ANNUITY": AMT_ANNUITY,
        "SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean": SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean,
        "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
        "SK_ID_CURR_DAYS_CREDIT_ENDDATE_max": SK_ID_CURR_DAYS_CREDIT_ENDDATE_max,
        "DAYS_EMPLOYED": DAYS_EMPLOYED, "CODE_GENDER": CODE_GENDER,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS, "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "CNT_CHILDREN": CNT_CHILDREN, "FLAG_OWN_CAR": FLAG_OWN_CAR,
        "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
        "OCCUPATION_TYPE": OCCUPATION_TYPE,
        "REGION_POPULATION_RELATIVE": REGION_POPULATION_RELATIVE,
        "HOUR_APPR_PROCESS_START": HOUR_APPR_PROCESS_START
    }

    if client_id_input != 0 and 'SK_ID_CURR' in st.session_state.client_data_form_values:
        full_client_data_from_id = {k: v for k, v in
                                    st.session_state.client_data_form_values.items() if
                                    k not in client_data_for_api}
        client_data_for_api.update(full_client_data_from_id)
        client_data_for_api['SK_ID_CURR'] = client_id_input

    st.subheader("Chargement et Calcul...")
    try:
        response = requests.post(API_URL, json=client_data_for_api)
        response.raise_for_status()
        result = response.json()

        prob_default = result.get("probability_default")
        pred_class = result.get("prediction_class")
        optimal_threshold_used = result.get("optimal_threshold_used")
        shap_values_raw = result.get("shap_values")
        shap_expected_value = result.get("shap_expected_value")

        st.success("Calcul terminé!")

        st.subheader("📈 Score de Crédit et Décision")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Probabilité de Défaut", value=f"{prob_default:.2%}")
            if prob_default is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=prob_default * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilité de défaut du client"},
                    gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, optimal_threshold_used * 100], 'color': "lightgreen"},
                               {'range': [optimal_threshold_used * 100, 100],
                                'color': "lightcoral"}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75,
                                         'value': optimal_threshold_used * 100}}))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True,
                                config={'displayModeBar': False})

                st.info(f"Le seuil de décision est de **{optimal_threshold_used:.2%}**.")
                if pred_class == 0:
                    st.success(
                        f"**Décision : CRÉDIT ACCORDÉ** (Probabilité de défaut : {prob_default:.2%}, inférieure au seuil)")
                else:
                    st.error(
                        f"**Décision : CRÉDIT REFUSÉ** (Probabilité de défaut : {prob_default:.2%}, supérieure ou égale au seuil)")

        with col2:
            st.subheader("Interprétation du Score (Facteurs influents)")
            if shap_values_raw and "error" not in shap_values_raw and shap_values_raw != {
                "info": "SHAP explainer non disponible ou non initialisé."}:
                shap_df = pd.DataFrame(shap_values_raw.items(), columns=['feature', 'shap_value'])
                shap_df['abs_shap_value'] = np.abs(shap_df['shap_value'])
                shap_df = shap_df.sort_values(by='abs_shap_value', ascending=False).head(10)
                shap_df['feature_display_name'] = shap_df['feature'].map(
                    FEATURE_DESCRIPTIONS).fillna(shap_df['feature'])

                fig_shap = px.bar(
                    shap_df, x='shap_value', y='feature_display_name', orientation='h',
                    title='Top 10 des facteurs influençant la décision pour ce client',
                    labels={'shap_value': 'Impact sur la probabilité de défaut (valeur SHAP)',
                            'feature_display_name': 'Caractéristique'},
                    color='shap_value', color_continuous_scale=px.colors.diverging.RdBu,
                    hover_data={'shap_value': ':.4f'})
                fig_shap.update_layout(
                    xaxis_title="Impact sur la probabilité de défaut (valeur SHAP)",
                    yaxis_title="Caractéristique",
                    height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_shap, use_container_width=True)

                st.markdown("""
                * Les barres **rouges** indiquent des facteurs qui **augmentent** la probabilité de défaut du client.
                * Les barres **bleues** indiquent des facteurs qui **diminuent** la probabilité de défaut du client.
                * Plus la barre est longue, plus l'impact du facteur est important sur la décision du modèle.
                """)
            else:
                st.warning(
                    "Impossible de calculer les contributions individuelles des features (SHAP values).")

        st.subheader("📑 Informations Descriptives Détaillées du Client")

        client_info_df = pd.DataFrame.from_dict(client_data_for_api, orient='index',
                                                columns=['Valeur'])
        client_info_df.index.name = 'Caractéristique'
        client_info_df.index = client_info_df.index.map(FEATURE_DESCRIPTIONS).fillna(
            client_info_df.index)

        display_df = client_info_df.copy()
        for idx, row in display_df.iterrows():
            if 'Âge du client' in idx and pd.api.types.is_numeric_dtype(row['Valeur']):
                display_df.loc[idx, 'Valeur'] = f"{round(abs(row['Valeur']) / 365.25)} ans"
            elif 'Ancienneté d\'emploi' in idx and pd.api.types.is_numeric_dtype(row['Valeur']):
                if row['Valeur'] == 365243:
                    display_df.loc[idx, 'Valeur'] = "Non employé"
                else:
                    display_df.loc[idx, 'Valeur'] = f"{round(abs(row['Valeur']) / 365.25)} ans"
            elif 'Date de fin maximale des crédits passés' in idx and pd.api.types.is_numeric_dtype(
                    row['Valeur']):
                if row['Valeur'] < 0:
                    display_df.loc[
                        idx, 'Valeur'] = f"Il y a {round(abs(row['Valeur']) / 365.25)} ans"
                elif row['Valeur'] > 0:
                    display_df.loc[idx, 'Valeur'] = f"Dans {round(row['Valeur'] / 365.25)} ans"
                else:
                    display_df.loc[idx, 'Valeur'] = "Aujourd'hui"

        st.dataframe(
            display_df.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}),
            use_container_width=True)
        st.write("---")

        st.subheader("🔬 Comparaison avec l'Ensemble des Clients")
        st.markdown(
            "Comparez les caractéristiques du client actuel avec la distribution de l'ensemble de notre base de données.")

        comparison_features = [col for col in df_ref.columns if
                               col in FEATURE_DESCRIPTIONS or col == "_AGE_YEARS" or col == "_EMPLOYED_YEARS_CAT"]

        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            selected_feature_hist_tech_name = st.selectbox(
                "Sélectionnez une caractéristique à comparer (Histogramme):",
                comparison_features,
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x,
                                                               x) if x != '_EMPLOYED_YEARS_CAT' else 'Ancienneté d\'emploi (catégories)'
            )

            if selected_feature_hist_tech_name:
                plot_x_axis = selected_feature_hist_tech_name
                client_value_for_plot_hist = None

                if selected_feature_hist_tech_name == "_AGE_YEARS":
                    client_value_for_plot_hist = np.abs(
                        client_data_for_api.get("DAYS_BIRTH")) / 365.25
                elif selected_feature_hist_tech_name == "_EMPLOYED_YEARS_CAT":
                    if client_data_for_api.get("DAYS_EMPLOYED") == 365243:
                        client_value_for_plot_hist = "Non-employé"
                    else:
                        client_value_for_plot_hist = f"{round(np.abs(client_data_for_api.get('DAYS_EMPLOYED')) / 365.25)} ans"
                    # Assurez-vous que la catégorie du client existe dans les catégories du df_ref pour _EMPLOYED_YEARS_CAT
                    if client_value_for_plot_hist not in df_ref[
                        '_EMPLOYED_YEARS_CAT'].cat.categories:
                        df_ref['_EMPLOYED_YEARS_CAT'] = df_ref[
                            '_EMPLOYED_YEARS_CAT'].cat.add_categories([client_value_for_plot_hist])
                else:
                    client_value_for_plot_hist = client_data_for_api.get(
                        selected_feature_hist_tech_name)

                fig_hist = px.histogram(df_ref, x=plot_x_axis,
                                        title=f"Distribution de '{FEATURE_DESCRIPTIONS.get(selected_feature_hist_tech_name, selected_feature_hist_tech_name)}' dans la base",
                                        marginal="box",
                                        color_discrete_sequence=px.colors.qualitative.Plotly)

                if client_value_for_plot_hist is not None:
                    if pd.api.types.is_numeric_dtype(df_ref[plot_x_axis]) and isinstance(
                            client_value_for_plot_hist, (int, float)):
                        fig_hist.add_vline(x=client_value_for_plot_hist, line_dash="dash",
                                           line_color="red",
                                           annotation_text=f"Client: {client_value_for_plot_hist:.2f}",
                                           annotation_position="top right")
                    elif selected_feature_hist_tech_name == "_EMPLOYED_YEARS_CAT":
                        fig_hist.add_annotation(
                            x=client_value_for_plot_hist, y=1, text=f"Client",
                            showarrow=True, arrowhead=2, arrowcolor="red", font=dict(color="red"),
                            yref="paper", xref="x")

                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Veuillez sélectionner une caractéristique pour l'histogramme.")

        with col_comp2:
            st.markdown("### Analyse Bi-variée :")
            feature_x_tech_name = st.selectbox(
                "Axe X : Sélectionnez la première caractéristique :",
                comparison_features,
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x,
                                                               x) if x != '_EMPLOYED_YEARS_CAT' else 'Ancienneté d\'emploi (catégories)',
                key="feature_x"
            )
            feature_y_tech_name = st.selectbox(
                "Axe Y : Sélectionnez la seconde caractéristique :",
                comparison_features,
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x,
                                                               x) if x != '_EMPLOYED_YEARS_CAT' else 'Ancienneté d\'emploi (catégories)',
                key="feature_y"
            )

            if feature_x_tech_name and feature_y_tech_name:
                df_temp_scatter = df_ref.copy()

                client_x_val_plot = None
                if feature_x_tech_name == "_AGE_YEARS":
                    client_x_val_plot = np.abs(client_data_for_api.get("DAYS_BIRTH")) / 365.25
                elif feature_x_tech_name == "_EMPLOYED_YEARS_CAT":
                    if client_data_for_api.get("DAYS_EMPLOYED") == 365243:
                        client_x_val_plot = "Non-employé"
                    else:
                        client_x_val_plot = f"{round(np.abs(client_data_for_api.get('DAYS_EMPLOYED')) / 365.25)} ans"
                    if client_x_val_plot not in df_temp_scatter[
                        '_EMPLOYED_YEARS_CAT'].cat.categories:
                        df_temp_scatter['_EMPLOYED_YEARS_CAT'] = df_temp_scatter[
                            '_EMPLOYED_YEARS_CAT'].cat.add_categories([client_x_val_plot])

                else:
                    client_x_val_plot = client_data_for_api.get(feature_x_tech_name)

                client_y_val_plot = None
                if feature_y_tech_name == "_AGE_YEARS":
                    client_y_val_plot = np.abs(client_data_for_api.get("DAYS_BIRTH")) / 365.25
                elif feature_y_tech_name == "_EMPLOYED_YEARS_CAT":
                    if client_data_for_api.get("DAYS_EMPLOYED") == 365243:
                        client_y_val_plot = "Non-employé"
                    else:
                        client_y_val_plot = f"{round(np.abs(client_data_for_api.get('DAYS_EMPLOYED')) / 365.25)} ans"
                    if client_y_val_plot not in df_temp_scatter[
                        '_EMPLOYED_YEARS_CAT'].cat.categories:
                        df_temp_scatter['_EMPLOYED_YEARS_CAT'] = df_temp_scatter[
                            '_EMPLOYED_YEARS_CAT'].cat.add_categories([client_y_val_plot])
                else:
                    client_y_val_plot = client_data_for_api.get(feature_y_tech_name)

                if feature_x_tech_name in df_temp_scatter.columns and feature_y_tech_name in df_temp_scatter.columns:
                    fig_scatter = px.scatter(
                        df_temp_scatter.dropna(subset=[feature_x_tech_name, feature_y_tech_name]),
                        x=feature_x_tech_name, y=feature_y_tech_name,
                        title=f"Relation entre '{FEATURE_DESCRIPTIONS.get(feature_x_tech_name, feature_x_tech_name) if feature_x_tech_name != '_EMPLOYED_YEARS_CAT' else 'Ancienneté d\'emploi (catégories)'}' et '{FEATURE_DESCRIPTIONS.get(feature_y_tech_name, feature_y_tech_name) if feature_y_tech_name != '_EMPLOYED_YEARS_CAT' else 'Ancienneté d\'emploi (catégories)'}'",
                        opacity=0.6,
                        hover_data={feature_x_tech_name: True, feature_y_tech_name: True},
                        color_discrete_sequence=px.colors.qualitative.Plotly)
                    if client_x_val_plot is not None and client_y_val_plot is not None:
                        x_is_numeric = pd.api.types.is_numeric_dtype(
                            df_temp_scatter[feature_x_tech_name])
                        y_is_numeric = pd.api.types.is_numeric_dtype(
                            df_temp_scatter[feature_y_tech_name])

                        if x_is_numeric and y_is_numeric and isinstance(client_x_val_plot, (
                        int, float)) and isinstance(client_y_val_plot, (int, float)):
                            fig_scatter.add_trace(go.Scatter(
                                x=[client_x_val_plot], y=[client_y_val_plot], mode='markers',
                                marker=dict(color='red', size=12, symbol='star'),
                                name='Client Actuel',
                                hovertemplate=f"Client X: {client_x_val_plot}<br>Client Y: {client_y_val_plot}"))

                    fig_scatter.update_layout(height=400,
                                              xaxis_title=FEATURE_DESCRIPTIONS.get(
                                                  feature_x_tech_name,
                                                  feature_x_tech_name) if feature_x_tech_name != '_EMPLOYED_YEARS_CAT' else 'Ancienneté d\'emploi (catégories)',
                                              yaxis_title=FEATURE_DESCRIPTIONS.get(
                                                  feature_y_tech_name,
                                                  feature_y_tech_name) if feature_y_tech_name != '_EMPLOYED_YEARS_CAT' else 'Ancienneté d\'emploi (catégories)')
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning(
                        "Impossible de générer le graphique bi-varié car une ou plusieurs caractéristiques n'ont pas pu être traitées.")
            else:
                st.warning("Veuillez sélectionner deux caractéristiques pour l'analyse bi-variée.")

        st.markdown("---")
        st.subheader("🔍 Autres Graphiques Pertinents")

        if 'AMT_CREDIT' in df_ref.columns and 'NAME_EDUCATION_TYPE' in df_ref.columns:
            df_temp_box = df_ref.copy()
            df_temp_box['NAME_EDUCATION_TYPE'] = df_temp_box['NAME_EDUCATION_TYPE'].astype(str)

            fig_box = px.box(df_temp_box, x="NAME_EDUCATION_TYPE", y="AMT_CREDIT",
                             title="Montant du crédit par niveau d'éducation",
                             labels={"NAME_EDUCATION_TYPE": FEATURE_DESCRIPTIONS.get(
                                 "NAME_EDUCATION_TYPE"),
                                     "AMT_CREDIT": FEATURE_DESCRIPTIONS.get("AMT_CREDIT")},
                             color="NAME_EDUCATION_TYPE",
                             color_discrete_map={"Higher education": "blue",
                                                 "Secondary / secondary special": "green",
                                                 "Incomplete higher": "orange",
                                                 "Lower secondary": "red",
                                                 "Academic degree": "purple", "nan": "gray"})
            client_edu = client_data_for_api.get('NAME_EDUCATION_TYPE')
            client_credit = client_data_for_api.get('AMT_CREDIT')
            if client_edu and client_credit is not None:
                fig_box.add_trace(go.Scatter(
                    x=[client_edu], y=[client_credit], mode='markers',
                    marker=dict(color='red', size=12, symbol='star'), name='Client Actuel',
                    hovertemplate=f"Client Éducation: {client_edu}<br>Client Crédit: {client_credit}"))
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Impossible de se connecter à l'API. Veuillez vérifier l'URL et que l'API est bien démarrée.")
    except requests.exceptions.Timeout:
        st.error("⏳ La requête a expiré. L'API est peut-être trop lente ou surchargée.")
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ Erreur HTTP de l'API : {e.response.status_code} - {e.response.text}")
    except json.JSONDecodeError:
        st.error(
            "🚫 Erreur de décodage JSON de la réponse de l'API. Vérifiez le format de la réponse.")
    except Exception as e:
        st.error(f"🔥 Une erreur inattendue est survenue : {e}")
        st.exception(e)
        st.code(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.header("Accessibilité (WCAG)")
st.sidebar.markdown("""
Ce dashboard prend en compte certains critères d'accessibilité :
* **Critère 1.1.1 Contenu non textuel :** Les graphiques Plotly génèrent des images SVG qui peuvent inclure des balises `<title>` et `<desc>` pour les lecteurs d'écran (support partiel par défaut de Plotly, améliorations possibles via des attributs alt-text si Streamlit le permet directement sur les graphiques).
* **Critère 1.4.1 Utilisation de la couleur :** Les informations ne sont pas transmises *uniquement* par la couleur (ex: seuil visible avec une ligne pointillée sur la jauge). Les palettes de couleurs sont choisies pour une meilleure différenciation.
* **Critère 1.4.3 Contraste (minimum) :** Les couleurs de texte et d'arrière-plan de Streamlit sont généralement conformes. Pour les graphiques, des palettes de couleurs contrastées sont utilisées (e.g., `px.colors.diverging.RdBu`).
* **Critère 1.4.4 Redimensionnement du texte :** Streamlit permet le redimensionnement du texte via les fonctions de zoom du navigateur.
* **Critère 2.4.2 Titre de page :** Le titre de la page est défini (`st.set_page_config`).
""")