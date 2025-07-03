import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap

# --- Dictionnaire de traduction des noms de variables (intégré) ---
FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_1": "Score Source Externe 1 (information financière externe)",
    "EXT_SOURCE_3": "Score Source Externe 3 (information financière externe)",
    "AMT_CREDIT": "Montant du crédit demandé",
    "DAYS_BIRTH": "Âge du client (en jours, négatif)",
    "EXT_SOURCE_2": "Score Source Externe 2 (information financière externe)",
    "AMT_ANNUITY": "Montant des annuités du prêt",
    "SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean": "Moyenne des échéances futures impayées (crédits précédents)",
    "DAYS_ID_PUBLISH": "Ancienneté de la dernière mise à jour de l'ID (en jours, négatif)",
    "SK_ID_CURR_DAYS_CREDIT_ENDDATE_max": "Date de fin maximale des crédits passés (en jours)",
    "DAYS_EMPLOYED": "Ancienneté d'emploi actuelle (en jours, négatif, 365243 si non employé)",

    "CODE_GENDER": "Genre",
    "NAME_EDUCATION_TYPE": "Niveau d'éducation",
    "NAME_FAMILY_STATUS": "Statut familial",
    "AMT_INCOME_TOTAL": "Revenu annuel total",
    "CNT_CHILDREN": "Nombre d'enfants",
    "FLAG_OWN_CAR": "Possède une voiture",
    "FLAG_OWN_REALTY": "Possède un bien immobilier",
    "OCCUPATION_TYPE": "Type d'emploi",
    "REGION_POPULATION_RELATIVE": "Densité de population de la région de résidence",
    "HOUR_APPR_PROCESS_START": "Heure de début de la demande",

    "SK_ID_CURR_AMT_GOODS_PRICE_mean": "Moyenne du prix des biens des anciens crédits",
    "SK_ID_CURR_AMT_PAYMENT_CURRENT_mean": "Moyenne des paiements actuels des crédits",
    "SK_ID_CURR_AMT_INSTALMENT_mean": "Moyenne des montants d'échéances des anciens crédits",
    "SK_ID_CURR_AMT_CREDIT_SUM_DEBT_sum": "Somme des dettes des anciens crédits",
    "SK_ID_CURR_AMT_ANNUITY_mean": "Moyenne des annuités des anciens crédits",
    "SK_ID_CURR_AMT_TOTAL_RECEIVABLE_sum": "Somme des montants totaux à recevoir des anciens crédits",
    "SK_ID_CURR_AMT_TOTAL_RECEIVABLE_max": "Maximum du montant total à recevoir des anciens crédits",
    "SK_ID_CURR_AMT_RECEIVABLE_PRINCIPAL_sum": "Somme du capital à recevoir des anciens crédits",
    "SK_ID_CURR_AMT_CREDIT_SUM_sum": "Somme des montants de crédit des anciens crédits",
    "SK_ID_CURR_CNT_INSTALMENT_mean": "Moyenne du nombre d'échéances des anciens crédits",
    "SK_ID_CURR_MONTHS_BALANCE_max_x": "Mois maximal de l'historique de bureau",
    "SK_ID_CURR_PAYMENT_DIFF_sum": "Somme des différences de paiement des anciens crédits",
    "SK_ID_CURR_CNT_INSTALMENT_max": "Maximum du nombre d'échéances des anciens crédits",
    "SK_ID_CURR_DBD_sum_x": "Somme des jours avant la date de paiement (bureau)",
    "SK_ID_CURR_DBD_max_y": "Maximum des jours avant la date de paiement (POS/Cash)",
    "SK_ID_CURR_AMT_CREDIT_LIMIT_ACTUAL_min": "Minimum du montant de la limite de crédit actuelle",
    "SK_ID_CURR_AMT_DRAWINGS_POS_CURRENT_mean": "Moyenne des retraits POS actuels",
    "SK_ID_CURR_SK_DPD_mean_x": "Moyenne des jours d'arriérés (bureau)",
    "SK_ID_CURR_CNT_DRAWINGS_POS_CURRENT_sum": "Somme des retraits POS actuels",
    "SK_ID_CURR_DBD_mean_y": "Moyenne des jours avant la date de paiement (POS/Cash)",
    "SK_ID_CURR_CNT_DRAWINGS_ATM_CURRENT_sum": "Somme des retraits ATM actuels",
    "SK_ID_CURR_AMT_CREDIT_SUM_DEBT_mean": "Moyenne de la dette des anciens crédits",
    "SK_ID_CURR_SK_DPD_mean_y": "Moyenne des jours d'arriérés par définition (POS/Cash)",
    "SK_ID_CURR_CNT_DRAWINGS_ATM_CURRENT_max": "Maximum des retraits ATM actuels",
    "SK_ID_CURR_MONTHS_BALANCE_min_y": "Mois minimal de l'historique de solde (POS/Cash)",
    "SK_ID_CURR_DPD_sum_y": "Somme des jours d'arriérés (POS/Cash)",

    "SK_ID_CURR_NAME_GOODS_CATEGORY_Medicine_mean": "Moyenne des prêts pour Médicaments",
    "SK_ID_CURR_NAME_TYPE_SUITE_Spouse_partner_mean": "Moyenne des clients accompagnés par conjoint/partenaire",
    "SK_ID_CURR_CHANNEL_TYPE_Stone_mean": "Moyenne des demandes via canal 'Stone'",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Refused_mean_y": "Moyenne des contrats refusés (précédentes applications)",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_XNA_mean": "Moyenne des contrats au statut non spécifié",
    "SK_ID_CURR_SK_DPD_DEF_max_x": "Maximum des jours d'arriérés par définition (bureau)",
    "SK_ID_CURR_CODE_REJECT_REASON_XNA_mean": "Moyenne des raisons de rejet non spécifiées",
    "SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_MONDAY_mean": "Moyenne des demandes commencées un lundi",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Other_mean": "Moyenne des prêts pour Autres biens",
    "SK_ID_CURR_CREDIT_TYPE_Car_loan_mean": "Moyenne des prêts automobiles",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Canceled_mean_x": "Moyenne des contrats annulés (bureau)",
    "SK_ID_CURR_CNT_DRAWINGS_CURRENT_mean": "Moyenne des retraits actuels",

    "SK_ID_CURR_CHANNEL_TYPE_Channel_of_corporate_sales_mean": "Moyenne des demandes via canal de ventes corporate",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_House_Construction_mean": "Moyenne des prêts pour Construction de maison",
    "SK_ID_CURR_CREDIT_TYPE_Mortgage_mean": "Moyenne des prêts hypothécaires",
    "SK_ID_CURR_NAME_YIELD_GROUP_low_action_mean": "Moyenne des groupes de rendement 'faible action'",
    "SK_ID_CURR_CODE_REJECT_REASON_XAP_mean": "Moyenne des rejets par XAP",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Jewelry_mean": "Moyenne des prêts pour Bijoux",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Sent_proposal_mean": "Moyenne des contrats avec proposition envoyée",
    "SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_THURSDAY_mean": "Moyenne des demandes commencées un jeudi",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_AudioVideo_mean": "Moyenne des prêts pour Audio/Vidéo",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Car_repairs_mean": "Moyenne des prêts pour Réparations automobiles",
    "SK_ID_CURR_CODE_REJECT_REASON_CLIENT_mean": "Moyenne des rejets par décision client",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Signed_mean_y": "Moyenne des contrats signés (précédentes applications)",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Homewares_mean": "Moyenne des prêts pour Articles ménagers",
    "SK_ID_CURR_NAME_PAYMENT_TYPE_Noncash_from_your_account_mean": "Moyenne des paiements non cash depuis compte",
    "SK_ID_CURR_NAME_CLIENT_TYPE_New_mean": "Moyenne des clients de type 'Nouveau'",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Active_mean_y": "Moyenne des contrats actifs (précédentes applications)",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Consumer_Electronics_mean": "Moyenne des prêts pour Électronique grand public",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Purchase_of_electronic_equipment_mean": "Moyenne des prêts pour Achat d'équipement électronique",
    "SK_ID_CURR_NAME_CLIENT_TYPE_XNA_mean": "Moyenne des clients de type non spécifié",
    "SK_ID_CURR_NAME_PRODUCT_TYPE_walkin_mean": "Moyenne des produits de type 'walk-in'",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Education_mean": "Moyenne des prêts pour Éducation",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Medicine_mean": "Moyenne des prêts pour Médicaments",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Completed_mean_y": "Moyenne des contrats complétés (précédentes applications)",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Photo__Cinema_Equipment_mean": "Moyenne des prêts pour Équipement photo/cinéma",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Connectivity_mean": "Moyenne des demandes via industrie 'Connectivité'",
    "SK_ID_CURR_NAME_PAYMENT_TYPE_Cashless_from_the_account_of_the_employer_mean": "Moyenne des paiements sans cash depuis compte employeur",
    "SK_ID_CURR_CREDIT_TYPE_Real_estate_loan_mean": "Moyenne des prêts immobiliers",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Auto_Accessories_mean": "Moyenne des prêts pour Accessoires auto",
    "SK_ID_CURR_CHANNEL_TYPE_Credit_and_cash_offices_mean": "Moyenne des demandes via bureaux de crédit et cash",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal_mean": "Moyenne des prêts avec refus de spécifier le but",
    "SK_ID_CURR_CREDIT_TYPE_Consumer_credit_mean": "Moyenne des crédits à la consommation",
    "SK_ID_CURR_CREDIT_TYPE_Interbank_credit_mean": "Moyenne des crédits interbancaires",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_XNA_mean": "Moyenne des prêts avec but non spécifié",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Urgent_needs_mean": "Moyenne des prêts pour Besoins urgents",
    "SK_ID_CURR_RATE_INTEREST_PRIVILEGED_mean": "Moyenne du taux d'intérêt privilégié",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Office_Appliances_mean": "Moyenne des prêts pour Appareils de bureau",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Wedding__gift__holiday_mean": "Moyenne des prêts pour Mariage, cadeau, vacances",
    "SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_TUESDAY_mean": "Moyenne des demandes commencées un mardi",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Approved_mean": "Moyenne des contrats approuvés",
    "SK_ID_CURR_CODE_REJECT_REASON_SCO_mean": "Moyenne des rejets par SCO",
    "SK_ID_CURR_CREDIT_ACTIVE_Sold_mean": "Moyenne des crédits 'vendus'",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Demand_mean_y": "Moyenne des contrats demandés (précédentes applications)",
    "SK_ID_CURR_NAME_PORTFOLIO_Cards_mean": "Moyenne des portefeuilles de type 'Cartes'",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Buying_a_garage_mean": "Moyenne des prêts pour Achat de garage",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Animals_mean": "Moyenne des prêts pour Animaux",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Clothing_and_Accessories_mean": "Moyenne des prêts pour Vêtements et accessoires",
    "SK_ID_CURR_CHANNEL_TYPE_Contact_center_mean": "Moyenne des demandes via centre d'appels",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Construction_Materials_mean": "Moyenne des prêts pour Matériaux de construction",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Buying_a_home_mean": "Moyenne des prêts pour Achat d'une maison",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_XAP_mean": "Moyenne des prêts avec but 'XAP'",
    "SK_ID_CURR_CREDIT_TYPE_Loan_for_working_capital_replenishment_mean": "Moyenne des prêts pour réapprovisionnement fonds de roulement",
    "SK_ID_CURR_CREDIT_TYPE_Another_type_of_loan_mean": "Moyenne des autres types de prêts",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Fitness_mean": "Moyenne des prêts pour Fitness",
    "SK_ID_CURR_NAME_PORTFOLIO_POS_mean": "Moyenne des portefeuilles de type 'POS'",
    "SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_SATURDAY_mean": "Moyenne des demandes commencées un samedi",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Gasification__water_supply_mean": "Moyenne des prêts pour Gazification, approvisionnement en eau",
    "SK_ID_CURR_NAME_YIELD_GROUP_high_mean": "Moyenne des groupes de rendement 'élevé'",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Gardening_mean": "Moyenne des prêts pour Jardinage",
    "SK_ID_CURR_FLAG_LAST_APPL_PER_CONTRACT_Y_mean": "Moyenne du flag 'dernière application par contrat' à Oui",
    "SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_mean": "Moyenne des demandes commencées un mercredi",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Industry_mean": "Moyenne des demandes via industrie 'Industrie'",
    "SK_ID_CURR_NAME_PRODUCT_TYPE_xsell_mean": "Moyenne des produits de type 'cross-sell'",
    "SK_ID_CURR_CODE_REJECT_REASON_LIMIT_mean": "Moyenne des rejets par limite",
    "SK_ID_CURR_CREDIT_TYPE_Loan_for_business_development_mean": "Moyenne des prêts pour développement commercial",
    "SK_ID_CURR_NAME_TYPE_SUITE_Unaccompanied_mean": "Moyenne des clients non accompagnés",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Additional_Service_mean": "Moyenne des prêts pour Services additionnels",
    "SK_ID_CURR_CHANNEL_TYPE_AP_Cash_loan_mean": "Moyenne des demandes via canal 'AP Cash loan'",
    "SK_ID_CURR_NAME_CONTRACT_TYPE_Cash_loans_mean": "Moyenne des contrats de type 'Prêts en espèces'",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_MLM_partners_mean": "Moyenne des demandes via industrie 'Partenaires MLM'",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Medical_Supplies_mean": "Moyenne des prêts pour Fournitures médicales",
    "SK_ID_CURR_CREDIT_TYPE_Cash_loan_nonearmarked_mean": "Moyenne des prêts en espèces non affectés",
    "SK_ID_CURR_CREDIT_ACTIVE_Closed_mean": "Moyenne des crédits 'fermés'",
    "SK_ID_CURR_NAME_CLIENT_TYPE_Repeater_mean": "Moyenne des clients de type 'Répéteur'",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Direct_Sales_mean": "Moyenne des prêts pour Ventes directes",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Computers_mean": "Moyenne des prêts pour Ordinateurs",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Clothing_mean": "Moyenne des demandes via industrie 'Vêtements'",
    "SK_ID_CURR_CREDIT_TYPE_Credit_card_mean": "Moyenne des crédits par carte",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Returned_to_the_store_mean": "Moyenne des contrats retournés au magasin",
    "SK_ID_CURR_NAME_TYPE_SUITE_Other_A_mean": "Moyenne des clients accompagnés par 'Autre A'",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Furniture_mean": "Moyenne des prêts pour Meubles",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Vehicles_mean": "Moyenne des prêts pour Véhicules",
    "SK_ID_CURR_NAME_YIELD_GROUP_middle_mean": "Moyenne des groupes de rendement 'moyen'",
    "YEARS_BEGINEXPLUATATION_AVG": "Moyenne des années de début d'exploitation",
    "SK_ID_CURR_NAME_CONTRACT_TYPE_Consumer_loans_mean": "Moyenne des contrats de type 'Crédits à la consommation'",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Repairs_mean": "Moyenne des prêts pour Réparations",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Building_a_house_or_an_annex_mean": "Moyenne des prêts pour Construction de maison ou annexe",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Hobby_mean": "Moyenne des prêts pour Loisirs",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Canceled_mean_y": "Moyenne des contrats annulés (précédentes applications)",
    "SK_ID_CURR_NAME_TYPE_SUITE_Other_B_mean": "Moyenne des clients accompagnés par 'Autre B'",
    "SK_ID_CURR_NAME_PORTFOLIO_Cars_mean": "Moyenne des portefeuilles de type 'Voitures'",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Furniture_mean": "Moyenne des prêts pour Meubles",
    "SK_ID_CURR_NAME_PRODUCT_TYPE_XNA_mean": "Moyenne des produits de type non spécifié",
    "SK_ID_CURR_NAME_TYPE_SUITE_Children_mean": "Moyenne des clients accompagnés par des enfants",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Auto_technology_mean": "Moyenne des demandes via industrie 'Technologie auto'",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Amortized_debt_mean": "Moyenne des dettes amorties",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Refused_mean_x": "Moyenne des contrats refusés (bureau)",
    "SK_ID_CURR_CREDIT_ACTIVE_Bad_debt_mean": "Moyenne des créances irrécouvrables",
    "SK_ID_CURR_NAME_YIELD_GROUP_low_normal_mean": "Moyenne des groupes de rendement 'faible normal'",
    "SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_FRIDAY_mean": "Moyenne des demandes commencées un vendredi",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Completed_mean_x": "Moyenne des contrats complétés (bureau)",
    "SK_ID_CURR_CHANNEL_TYPE_Car_dealer_mean": "Moyenne des demandes via concessionnaire auto",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Construction_mean": "Moyenne des demandes via industrie 'Construction'",
    "SK_ID_CURR_NAME_CONTRACT_TYPE_XNA_mean": "Moyenne des contrats de type non spécifié",
    "SK_ID_CURR_CHANNEL_TYPE_Countrywide_mean": "Moyenne des demandes via canal 'National'",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Signed_mean_x": "Moyenne des contrats signés (bureau)",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Active_mean_x": "Moyenne des contrats actifs (bureau)",
    "SK_ID_CURR_NAME_PORTFOLIO_Cash_mean": "Moyenne des portefeuilles de type 'Cash'",
    "SK_ID_CURR_NAME_YIELD_GROUP_XNA_mean": "Moyenne des groupes de rendement non spécifiés",
    "SK_ID_CURR_CREDIT_TYPE_Microloan_mean": "Moyenne des microcrédits",
    "SK_ID_CURR_CHANNEL_TYPE_Regional__Local_mean": "Moyenne des demandes via canal 'Régional/Local'",
    "SK_ID_CURR_CREDIT_TYPE_Loan_for_the_purchase_of_equipment_mean": "Moyenne des prêts pour l'achat d'équipement",
    "SK_ID_CURR_CODE_REJECT_REASON_HC_mean": "Moyenne des rejets par 'HC'",
    "SK_ID_CURR_NAME_TYPE_SUITE_Group_of_people_mean": "Moyenne des clients accompagnés par un groupe",
    "SK_ID_CURR_CODE_REJECT_REASON_SCOFR_mean": "Moyenne des rejets par 'SCOFR'",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Everyday_expenses_mean": "Moyenne des prêts pour Dépenses quotidiennes",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Tourism_mean": "Moyenne des prêts pour Tourisme",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Mobile_mean": "Moyenne des prêts pour Mobile",
    "SK_ID_CURR_CREDIT_ACTIVE_Active_mean": "Moyenne des crédits 'actifs'",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Money_for_a_third_person_mean": "Moyenne des prêts pour Argent pour une tierce personne",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Other_mean": "Moyenne des prêts pour Autres buts",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Tourism_mean": "Moyenne des demandes via industrie 'Tourisme'",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Insurance_mean": "Moyenne des prêts pour Assurance",
    "SK_ID_CURR_CREDIT_TYPE_Loan_for_purchase_of_shares_margin_lending_mean": "Moyenne des prêts pour achat d'actions/marge",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Business_development_mean": "Moyenne des prêts pour Développement commercial",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Unused_offer_mean": "Moyenne des offres non utilisées",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_XNA_mean": "Moyenne des prêts pour biens non spécifiés",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Furniture_mean": "Moyenne des demandes via industrie 'Meubles'",
    "SK_ID_CURR_CREDIT_TYPE_Mobile_operator_loan_mean": "Moyenne des prêts d'opérateur mobile",
    "SK_ID_CURR_CODE_REJECT_REASON_SYSTEM_mean": "Moyenne des rejets par système",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Education_mean": "Moyenne des prêts pour Éducation",
    "SK_ID_CURR_WEEKDAY_APPR_PROCESS_START_SUNDAY_mean": "Moyenne des demandes commencées un dimanche",
    "SK_ID_CURR_NAME_PAYMENT_TYPE_Cash_through_the_bank_mean": "Moyenne des paiements en espèces via banque",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Payments_on_other_loans_mean": "Moyenne des prêts pour Paiements sur autres prêts",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Sport_and_Leisure_mean": "Moyenne des prêts pour Sport et Loisirs",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Approved_mean_y": "Moyenne des contrats approuvés (précédentes applications)",
    "SK_ID_CURR_CREDIT_TYPE_Unknown_type_of_loan_mean": "Moyenne des types de prêts inconnus",
    "SK_ID_CURR_NAME_PAYMENT_TYPE_XNA_mean": "Moyenne des types de paiement non spécifiés",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Buying_a_used_car_mean": "Moyenne des prêts pour Achat de voiture d'occasion",
    "SK_ID_CURR_NAME_CONTRACT_STATUS_Demand_mean_x": "Moyenne des contrats demandés (bureau)",
    "SK_ID_CURR_NAME_TYPE_SUITE_Family_mean": "Moyenne des clients accompagnés par famille",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_XNA_mean": "Moyenne des demandes via industrie non spécifiée",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Buying_a_new_car_mean": "Moyenne des prêts pour Achat de voiture neuve",
    "SK_ID_CURR_NAME_CONTRACT_TYPE_Revolving_loans_mean": "Moyenne des prêts revolving",
    "SK_ID_CURR_NAME_GOODS_CATEGORY_Weapon_mean": "Moyenne des prêts pour Armes",
    "SK_ID_CURR_CODE_REJECT_REASON_VERIF_mean": "Moyenne des rejets par vérification",
    "SK_ID_CURR_NAME_PORTFOLIO_XNA_mean": "Moyenne des portefeuilles de type non spécifié",
    "SK_ID_CURR_NAME_CASH_LOAN_PURPOSE_Journey_mean": "Moyenne des prêts pour Voyage",
    "SK_ID_CURR_NAME_SELLER_INDUSTRY_Consumer_electronics_mean": "Moyenne des demandes via industrie 'Électronique grand public'",
    "SK_ID_CURR_AMT_CREDIT_mean": "Moyenne du montant du crédit des anciens crédits",
    "SK_ID_CURR_AMT_DRAWINGS_CURRENT_mean": "Moyenne des retraits actuels",

    "NAME_CONTRACT_TYPE": "Type de contrat",
    "NAME_TYPE_SUITE": "Type d'accompagnement de la demande",
    "NAME_INCOME_TYPE": "Type de revenu",
    "NAME_HOUSING_TYPE": "Type de logement",
    "WEEKDAY_APPR_PROCESS_START": "Jour de la semaine de la demande",
    "ORGANIZATION_TYPE": "Type d'organisation de l'emploi",
    "FONDKAPREMONT_MODE": "Mode de financement de la réparation capitale",
    "HOUSETYPE_MODE": "Type de maison",
    "WALLSMATERIAL_MODE": "Matériau des murs",
    "EMERGENCYSTATE_MODE": "État d'urgence du bâtiment",
    "REGION_RATING_CLIENT": "Notation de la région (par le client)",
    "REGION_RATING_CLIENT_W_CITY": "Notation de la région (par le client avec ville)",
}

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Prêt à dépenser : Outil de Scoring Crédit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- URL de l'API ---
API_URL = "https://ilkan77-openclassroom.hf.space/predict"


# --- Chargement des données d'exemple pour les comparaisons et le pré-remplissage ---
@st.cache_data
def load_full_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Supprimer la colonne TARGET si elle existe pour la prédiction
        if 'TARGET' in df.columns:
            df = df.drop(columns=['TARGET'])
        return df
    except FileNotFoundError:
        st.error(
            f"Fichier '{file_path}' non trouvé. Assurez-vous qu'il est dans le même répertoire que l'application Streamlit.")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur


# Charger le DataFrame complet
df_full = load_full_data("application_train.csv")  # Assurez-vous que c'est le bon nom de fichier

# Filtrer df_ref pour n'avoir que les colonnes pertinentes pour les comparaisons globales
# et prendre un échantillon pour des raisons de performance si df_full est très grand
relevant_cols_for_sample = list(FEATURE_DESCRIPTIONS.keys())
df_ref = df_full[df_full.columns.intersection(relevant_cols_for_sample)].sample(
    min(1000, len(df_full)), random_state=42)

# --- Titre de l'application ---
st.title("📊 Prêt à dépenser : Outil de Scoring Crédit pour les Chargés de Clientèle")

st.markdown("""
Bienvenue sur le dashboard interactif d'aide à la décision d'octroi de crédit.
Cet outil vous permet de visualiser le score de crédit d'un client,
sa probabilité de défaut, et les facteurs qui ont influencé cette décision.
Vous pouvez également comparer le profil du client avec l'ensemble de la base.
""")

# --- Sidebar pour les informations client ---
st.sidebar.header("👤 Informations Client Actuel")
st.sidebar.markdown(
    "Saisissez un ID client pour pré-remplir les champs, ou entrez les informations manuellement.")

# Champ pour l'ID client
client_id_input = st.sidebar.number_input("ID Client (Ex: 100002)", min_value=0, value=0, step=1,
                                          help="Saisissez l'ID d'un client existant pour charger ses données.")
load_client_data_button = st.sidebar.button("Charger données client par ID")

# Initialisation des valeurs par défaut des champs du formulaire
# Utilisation de st.session_state pour persister les valeurs des champs
if 'client_data_form_values' not in st.session_state:
    st.session_state.client_data_form_values = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_3": 0.5,
        "AMT_CREDIT": 250000.0,
        "DAYS_BIRTH": -15000,
        "EXT_SOURCE_2": 0.5,
        "AMT_ANNUITY": 25000.0,
        "SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean": 0.0,
        "DAYS_ID_PUBLISH": -1000,
        "SK_ID_CURR_DAYS_CREDIT_ENDDATE_max": 0.0,
        "DAYS_EMPLOYED": -2000,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "AMT_INCOME_TOTAL": 150000.0,
        "CNT_CHILDREN": 0,
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "OCCUPATION_TYPE": "Laborers",
        "REGION_POPULATION_RELATIVE": 0.018801,
        "HOUR_APPR_PROCESS_START": 12,
    }

# Logique pour ajouter un client par ID
if load_client_data_button and client_id_input != 0:
    client_row = df_full[df_full['SK_ID_CURR'] == client_id_input]
    if not client_row.empty:
        st.sidebar.success(f"Données pour l'ID {client_id_input} chargées.")
        client_data = client_row.iloc[0].to_dict()

        for feature, value in client_data.items():
            if feature in st.session_state.client_data_form_values:
                if pd.isna(value):
                    if isinstance(st.session_state.client_data_form_values[feature], (int, float)):
                        st.session_state.client_data_form_values[feature] = 0.0
                    elif isinstance(st.session_state.client_data_form_values[feature], str):
                        st.session_state.client_data_form_values[feature] = "XNA" if feature in [
                            "CODE_GENDER",
                            "OCCUPATION_TYPE"] else "Unknown"
                else:
                    st.session_state.client_data_form_values[feature] = value
            elif feature not in st.session_state.client_data_form_values and feature != 'SK_ID_CURR':
                st.session_state.client_data_form_values[feature] = value
    else:
        st.sidebar.warning(f"ID Client {client_id_input} non trouvé dans la base de données.")
        st.session_state.client_data_form_values = {
            "EXT_SOURCE_1": 0.5,  # Reset to default if ID not found
            "EXT_SOURCE_3": 0.5,
            "AMT_CREDIT": 250000.0,
            "DAYS_BIRTH": -15000,
            "EXT_SOURCE_2": 0.5,
            "AMT_ANNUITY": 25000.0,
            "SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean": 0.0,
            "DAYS_ID_PUBLISH": -1000,
            "SK_ID_CURR_DAYS_CREDIT_ENDDATE_max": 0.0,
            "DAYS_EMPLOYED": -2000,
            "CODE_GENDER": "M",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Married",
            "AMT_INCOME_TOTAL": 150000.0,
            "CNT_CHILDREN": 0,
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "Y",
            "OCCUPATION_TYPE": "Laborers",
            "REGION_POPULATION_RELATIVE": 0.018801,
            "HOUR_APPR_PROCESS_START": 12,
        }

with st.sidebar.form("client_data_form"):
    st.markdown("### Champs essentiels pour le calcul du score:")

    # Numeric inputs with session state for initial values
    EXT_SOURCE_1 = st.number_input(
        FEATURE_DESCRIPTIONS.get("EXT_SOURCE_1", "Score Source Externe 1"),
        value=st.session_state.client_data_form_values.get("EXT_SOURCE_1"), format="%.6f",
        min_value=0.0, max_value=1.0,
        help="Score normalisé d'une source de données externe (plus élevé = meilleur).")
    EXT_SOURCE_3 = st.number_input(
        FEATURE_DESCRIPTIONS.get("EXT_SOURCE_3", "Score Source Externe 3"),
        value=st.session_state.client_data_form_values.get("EXT_SOURCE_3"), format="%.6f",
        min_value=0.0, max_value=1.0, help="Score normalisé d'une autre source externe.")
    AMT_CREDIT = st.number_input(
        FEATURE_DESCRIPTIONS.get("AMT_CREDIT", "Montant du crédit demandé"),
        value=st.session_state.client_data_form_values.get("AMT_CREDIT"), min_value=0.0,
        max_value=5000000.0, help="Montant total du crédit demandé par le client.")

    days_birth_val = st.session_state.client_data_form_values.get("DAYS_BIRTH")
    days_birth_input = st.number_input("Âge du client (en jours, ex: -15000)", value=int(
        days_birth_val) if days_birth_val is not None else -15000, min_value=-30000,
                                       max_value=-7000,
                                       help="Âge du client au moment de la demande, en jours (valeurs négatives : nombre de jours depuis la naissance).")
    st.info(f"Soit environ {round(abs(days_birth_input) / 365.25)} ans.")
    DAYS_BIRTH = days_birth_input

    EXT_SOURCE_2 = st.number_input(
        FEATURE_DESCRIPTIONS.get("EXT_SOURCE_2", "Score Source Externe 2"),
        value=st.session_state.client_data_form_values.get("EXT_SOURCE_2"), format="%.6f",
        min_value=0.0, max_value=1.0, help="Score normalisé d'une troisième source externe.")
    AMT_ANNUITY = st.number_input(
        FEATURE_DESCRIPTIONS.get("AMT_ANNUITY", "Montant des annuités du prêt"),
        value=st.session_state.client_data_form_values.get("AMT_ANNUITY"), min_value=0.0,
        max_value=200000.0, help="Montant de l'annuité du prêt (versements annuels).")
    SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean = st.number_input(
        FEATURE_DESCRIPTIONS.get("SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean",
                                 "Moyenne des échéances futures impayées"),
        value=st.session_state.client_data_form_values.get("SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean"),
        min_value=0.0, max_value=100.0,
        help="Nombre moyen de versements futurs pour les crédits précédents du client.")

    days_id_publish_val = st.session_state.client_data_form_values.get("DAYS_ID_PUBLISH")
    days_id_publish_input = st.number_input("Ancienneté mise à jour ID (en jours, ex: -1000)",
                                            value=int(
                                                days_id_publish_val) if days_id_publish_val is not None else -1000,
                                            min_value=-10000, max_value=-1,
                                            help="Nombre de jours depuis la dernière publication/mise à jour de l'ID client (négatif).")
    st.info(f"Soit environ {round(abs(days_id_publish_input) / 365.25)} ans.")
    DAYS_ID_PUBLISH = days_id_publish_input

    sk_id_curr_days_credit_enddate_max_val = st.session_state.client_data_form_values.get(
        "SK_ID_CURR_DAYS_CREDIT_ENDDATE_max")
    sk_id_curr_days_credit_enddate_max_input = st.number_input(
        "Date fin max crédits passés (en jours, ex: 0.0)", value=float(
            sk_id_curr_days_credit_enddate_max_val) if sk_id_curr_days_credit_enddate_max_val is not None else 0.0,
        min_value=-10000.0, max_value=10000.0,
        help="Nombre maximal de jours entre la date actuelle et la date de fin prévue du crédit le plus récent du client (- = passé, + = futur).")
    if sk_id_curr_days_credit_enddate_max_input < 0:
        st.info(
            f"Date de fin du crédit le plus récent : il y a environ {round(abs(sk_id_curr_days_credit_enddate_max_input) / 365.25)} ans.")
    elif sk_id_curr_days_credit_enddate_max_input > 0:
        st.info(
            f"Date de fin du crédit le plus récent : dans environ {round(abs(sk_id_curr_days_credit_enddate_max_input) / 365.25)} ans.")
    else:
        st.info("Date de fin du crédit le plus récent : aujourd'hui.")
    SK_ID_CURR_DAYS_CREDIT_ENDDATE_max = sk_id_curr_days_credit_enddate_max_input

    days_employed_val = st.session_state.client_data_form_values.get("DAYS_EMPLOYED")
    days_employed_input = st.number_input("Ancienneté d'emploi (en jours, ex: -2000 ou 365243)",
                                          value=int(
                                              days_employed_val) if days_employed_val is not None else -2000,
                                          min_value=-20000, max_value=365243,
                                          help="Nombre de jours depuis le début de l'emploi actuel (négatif). Utilisez 365243 si le client est non-employé.")
    if days_employed_input == 365243:
        st.info("Client actuellement non-employé.")
    elif days_employed_input < 0:
        st.info(f"Ancienneté d'emploi : environ {round(abs(days_employed_input) / 365.25)} ans.")
    else:
        st.info("Valeur d'ancienneté d'emploi non standard (positive).")
    DAYS_EMPLOYED = days_employed_input

    st.markdown("### Autres informations descriptives (pour le profil client):")
    CODE_GENDER = st.selectbox(FEATURE_DESCRIPTIONS.get("CODE_GENDER", "Genre"), ["M", "F", "XNA"],
                               index=["M", "F", "XNA"].index(
                                   st.session_state.client_data_form_values.get("CODE_GENDER",
                                                                                "M")),
                               help="Genre du client (M: Masculin, F: Féminin, XNA: Non spécifié).")
    NAME_EDUCATION_TYPE = st.selectbox(
        FEATURE_DESCRIPTIONS.get("NAME_EDUCATION_TYPE", "Niveau d'éducation"),
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        index=["Secondary / secondary special", "Higher education", "Incomplete higher",
               "Lower secondary", "Academic degree"].index(
            st.session_state.client_data_form_values.get("NAME_EDUCATION_TYPE",
                                                         "Secondary / secondary special")),
        help="Plus haut niveau d'éducation atteint par le client.")
    NAME_FAMILY_STATUS = st.selectbox(
        FEATURE_DESCRIPTIONS.get("NAME_FAMILY_STATUS", "Statut familial"),
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        index=["Married", "Single / not married", "Civil marriage", "Separated", "Widow"].index(
            st.session_state.client_data_form_values.get("NAME_FAMILY_STATUS", "Married")),
        help="Statut marital du client.")
    AMT_INCOME_TOTAL = st.number_input(
        FEATURE_DESCRIPTIONS.get("AMT_INCOME_TOTAL", "Revenu annuel total"),
        value=st.session_state.client_data_form_values.get("AMT_INCOME_TOTAL"), min_value=0.0,
        max_value=5000000.0, help="Revenu total annuel du client.")
    CNT_CHILDREN = st.number_input(FEATURE_DESCRIPTIONS.get("CNT_CHILDREN", "Nombre d'enfants"),
                                   value=st.session_state.client_data_form_values.get(
                                       "CNT_CHILDREN"), min_value=0, max_value=20, step=1,
                                   help="Nombre d'enfants à charge.")
    FLAG_OWN_CAR = st.radio(FEATURE_DESCRIPTIONS.get("FLAG_OWN_CAR", "Possède une voiture"),
                            ["Y", "N"], index=["Y", "N"].index(
            st.session_state.client_data_form_values.get("FLAG_OWN_CAR", "N")), horizontal=True,
                            help="Le client possède-t-il une voiture ?")
    FLAG_OWN_REALTY = st.radio(
        FEATURE_DESCRIPTIONS.get("FLAG_OWN_REALTY", "Possède un bien immobilier"), ["Y", "N"],
        index=["Y", "N"].index(
            st.session_state.client_data_form_values.get("FLAG_OWN_REALTY", "Y")), horizontal=True,
        help="Le client possède-t-il un bien immobilier ?")
    OCCUPATION_TYPE = st.selectbox(FEATURE_DESCRIPTIONS.get("OCCUPATION_TYPE", "Type d'emploi"), [
        "Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff",
        "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff",
        "Security staff", "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers",
        "Realty agents", "Secretaries", "IT staff", "HR staff", "nan"
    ], index=[
        "Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff",
        "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff",
        "Security staff", "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers",
        "Realty agents", "Secretaries", "IT staff", "HR staff", "nan"
    ].index(st.session_state.client_data_form_values.get("OCCUPATION_TYPE", "Laborers")),
                                   help="Type d'emploi du client.")
    REGION_POPULATION_RELATIVE = st.number_input(
        FEATURE_DESCRIPTIONS.get("REGION_POPULATION_RELATIVE",
                                 "Densité de population de la région"),
        value=st.session_state.client_data_form_values.get("REGION_POPULATION_RELATIVE"),
        format="%.6f", min_value=0.0, max_value=1.0,
        help="Score normalisé de la population de la région de résidence (plus élevé = plus peuplé).")
    HOUR_APPR_PROCESS_START = st.number_input(
        FEATURE_DESCRIPTIONS.get("HOUR_APPR_PROCESS_START", "Heure de début de la demande"),
        value=st.session_state.client_data_form_values.get("HOUR_APPR_PROCESS_START"), min_value=0,
        max_value=23, step=1, help="Heure à laquelle la demande de prêt a commencé (format 24h).")

    submitted = st.form_submit_button("Calculer et Expliquer le Score")

# --- Affichage des résultats dans la zone principale ---
if submitted:
    # Prepare client data for API
    client_data_for_api = {
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "AMT_CREDIT": AMT_CREDIT,
        "DAYS_BIRTH": DAYS_BIRTH,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "AMT_ANNUITY": AMT_ANNUITY,
        "SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean": SK_ID_CURR_CNT_INSTALMENT_FUTURE_mean,
        "DAYS_ID_PUBLISH": DAYS_ID_PUBLISH,
        "SK_ID_CURR_DAYS_CREDIT_ENDDATE_max": SK_ID_CURR_DAYS_CREDIT_ENDDATE_max,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "CODE_GENDER": CODE_GENDER,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "CNT_CHILDREN": CNT_CHILDREN,
        "FLAG_OWN_CAR": FLAG_OWN_CAR,
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

        # --- Affichage du Score et de la Probabilité ---
        st.subheader("📈 Score de Crédit et Décision")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Probabilité de Défaut", value=f"{prob_default:.2%}")
            if prob_default is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_default * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilité de défaut du client"},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, optimal_threshold_used * 100], 'color': "lightgreen"},
                            {'range': [optimal_threshold_used * 100, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': optimal_threshold_used * 100
                        }
                    }
                ))
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

                # Map feature names to French descriptions
                shap_df['feature_display_name'] = shap_df['feature'].map(
                    FEATURE_DESCRIPTIONS).fillna(shap_df['feature'])

                fig_shap = px.bar(
                    shap_df,
                    x='shap_value',
                    y='feature_display_name',
                    orientation='h',
                    title='Top 10 des facteurs influençant la décision pour ce client',
                    labels={'shap_value': 'Impact sur la probabilité de défaut (valeur SHAP)',
                            'feature_display_name': 'Caractéristique'},
                    color='shap_value',
                    color_continuous_scale=px.colors.diverging.RdBu,
                    hover_data={'shap_value': ':.4f'}
                )
                fig_shap.update_layout(
                    xaxis_title="Impact sur la probabilité de défaut (valeur SHAP)",
                    yaxis_title="Caractéristique",
                    height=400,
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                st.markdown("""
                * Les barres **rouges** indiquent des facteurs qui **augmentent** la probabilité de défaut du client.
                * Les barres **bleues** indiquent des facteurs qui **diminuent** la probabilité de défaut du client.
                * Plus la barre est longue, plus l'impact du facteur est important sur la décision du modèle.
                """)
            else:
                st.warning(
                    "Impossible de calculer les contributions individuelles des features (SHAP values). Cela peut être dû à un problème avec l'initialisation de l'explainer SHAP dans l'API.")

        # --- Informations descriptives du client ---
        st.subheader("📑 Informations Descriptives Détaillées du Client")
        client_profile_display_data = {}
        data_to_display = client_data_for_api.copy()
        for k, v in data_to_display.items():
            if k == 'SK_ID_CURR':
                continue
            display_name = FEATURE_DESCRIPTIONS.get(k, k)
            if k == "DAYS_BIRTH":
                client_profile_display_data[display_name] = f"{round(abs(v) / 365.25)} ans"
            elif k == "DAYS_EMPLOYED":
                client_profile_display_data[
                    display_name] = "Non employé" if v == 365243 else f"{round(abs(v) / 365.25)} ans"
            elif k == "DAYS_ID_PUBLISH":
                client_profile_display_data[display_name] = f"Il y a {round(abs(v) / 365.25)} ans"
            elif k == "SK_ID_CURR_DAYS_CREDIT_ENDDATE_max":
                if v < 0:
                    client_profile_display_data[
                        display_name] = f"Il y a {round(abs(v) / 365.25)} ans"
                elif v > 0:
                    client_profile_display_data[display_name] = f"Dans {round(v / 365.25)} ans"
                else:
                    client_profile_display_data[display_name] = "Aujourd'hui"
            else:
                client_profile_display_data[display_name] = v

        st.dataframe(pd.DataFrame([client_profile_display_data]).T.rename(
            columns={0: 'Valeur Client'}).style.set_properties(
            **{'background-color': '#f0f2f6', 'color': 'black'}), use_container_width=True)
        st.write("---")

        # --- Comparaison avec l'ensemble des clients ---
        st.subheader("🔬 Comparaison avec l'Ensemble des Clients")
        st.markdown(
            "Comparez les caractéristiques du client actuel avec la distribution de l'ensemble de notre base de données.")

        col_comp1, col_comp2 = st.columns(2)

        with col_comp1:
            selected_feature_hist_tech_name = st.selectbox(
                "Sélectionnez une caractéristique à comparer (Histogramme):",
                [col for col in df_ref.columns if col in FEATURE_DESCRIPTIONS],
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x)
            )

            if selected_feature_hist_tech_name and selected_feature_hist_tech_name in df_ref.columns:
                df_temp = df_ref.copy()
                client_value_for_plot = client_data_for_api.get(selected_feature_hist_tech_name)

                if selected_feature_hist_tech_name == "DAYS_BIRTH":
                    df_temp["Âge (années)"] = np.abs(df_temp["DAYS_BIRTH"]) / 365.25
                    plot_x_axis = "Âge (années)"
                    if client_value_for_plot is not None:
                        client_value_for_plot = np.abs(client_value_for_plot) / 365.25
                elif selected_feature_hist_tech_name == "DAYS_EMPLOYED":
                    df_temp["Ancienneté d'emploi (années)"] = np.abs(
                        df_temp["DAYS_EMPLOYED"]) / 365.25
                    df_temp.loc[df_temp[
                                    "DAYS_EMPLOYED"] == 365243, "Ancienneté d'emploi (années)"] = "Non-employé"
                    plot_x_axis = "Ancienneté d'emploi (années)"
                    if client_value_for_plot is not None:
                        if client_value_for_plot == 365243:
                            client_value_for_plot = "Non-employé"
                        else:
                            client_value_for_plot = np.abs(client_value_for_plot) / 365.25
                else:
                    plot_x_axis = selected_feature_hist_tech_name

                fig_hist = px.histogram(df_temp, x=plot_x_axis,
                                        title=f"Distribution de '{FEATURE_DESCRIPTIONS.get(selected_feature_hist_tech_name, selected_feature_hist_tech_name)}' dans la base",
                                        marginal="box",
                                        color_discrete_sequence=px.colors.qualitative.Plotly
                                        )

                if client_value_for_plot is not None:
                    if pd.api.types.is_numeric_dtype(df_temp[plot_x_axis]) and isinstance(
                            client_value_for_plot, (int, float)):
                        fig_hist.add_vline(x=client_value_for_plot, line_dash="dash",
                                           line_color="red",
                                           annotation_text=f"Client: {client_value_for_plot:.2f}",
                                           annotation_position="top right")
                    elif isinstance(client_value_for_plot, str):
                        fig_hist.add_annotation(
                            x=client_value_for_plot,
                            y=1,
                            text=f"Client",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="red",
                            font=dict(color="red"),
                            yref="paper",
                            xref="x"
                        )

                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning(
                    "Caractéristique non trouvée pour l'histogramme dans les données de référence.")

        with col_comp2:
            st.markdown("### Analyse Bi-variée :")
            feature_x_tech_name = st.selectbox(
                "Axe X : Sélectionnez la première caractéristique :",
                [col for col in df_ref.columns if col in FEATURE_DESCRIPTIONS],
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x),
                key="feature_x"
            )
            feature_y_tech_name = st.selectbox(
                "Axe Y : Sélectionnez la seconde caractéristique :",
                [col for col in df_ref.columns if col in FEATURE_DESCRIPTIONS],
                format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x),
                key="feature_y"
            )

            if feature_x_tech_name and feature_y_tech_name and feature_x_tech_name in df_ref.columns and feature_y_tech_name in df_ref.columns:
                df_temp_scatter = df_ref.copy()
                client_x_val = client_data_for_api.get(feature_x_tech_name)
                client_y_val = client_data_for_api.get(feature_y_tech_name)

                if feature_x_tech_name in df_temp_scatter.columns:
                    if feature_x_tech_name == "DAYS_BIRTH":
                        df_temp_scatter["_X_Axis_"] = np.abs(df_temp_scatter["DAYS_BIRTH"]) / 365.25
                        if client_x_val is not None: client_x_val = np.abs(client_x_val) / 365.25
                    elif feature_x_tech_name == "DAYS_EMPLOYED":
                        df_temp_scatter["_X_Axis_"] = np.abs(
                            df_temp_scatter["DAYS_EMPLOYED"]) / 365.25
                        df_temp_scatter.loc[
                            df_temp_scatter["DAYS_EMPLOYED"] == 365243, "_X_Axis_"] = "Non-employé"
                        if client_x_val is not None: client_x_val = "Non-employé" if client_x_val == 365243 else np.abs(
                            client_x_val) / 365.25
                    else:
                        df_temp_scatter["_X_Axis_"] = df_temp_scatter[feature_x_tech_name]
                else:
                    df_temp_scatter[
                        "_X_Axis_"] = pd.NA
                    st.warning(
                        f"La colonne '{feature_x_tech_name}' n'est pas présente dans les données de référence pour l'axe X.")

                if feature_y_tech_name in df_temp_scatter.columns:
                    if feature_y_tech_name == "DAYS_BIRTH":
                        df_temp_scatter["_Y_Axis_"] = np.abs(df_temp_scatter["DAYS_BIRTH"]) / 365.25
                        if client_y_val is not None: client_y_val = np.abs(client_y_val) / 365.25
                    elif feature_y_tech_name == "DAYS_EMPLOYED":
                        df_temp_scatter["_Y_Axis_"] = np.abs(
                            df_temp_scatter["DAYS_EMPLOYED"]) / 365.25
                        df_temp_scatter.loc[
                            df_temp_scatter["DAYS_EMPLOYED"] == 365243, "_Y_Axis_"] = "Non-employé"
                        if client_y_val is not None: client_y_val = "Non-employé" if client_y_val == 365243 else np.abs(
                            client_y_val) / 365.25
                    else:
                        df_temp_scatter["_Y_Axis_"] = df_temp_scatter[feature_y_tech_name]
                else:
                    df_temp_scatter[
                        "_Y_Axis_"] = pd.NA
                    st.warning(
                        f"La colonne '{feature_y_tech_name}' n'est pas présente dans les données de référence pour l'axe Y.")

                # Proceed only if both columns were found and assigned for plotting
                if "_X_Axis_" in df_temp_scatter.columns and "_Y_Axis_" in df_temp_scatter.columns:
                    fig_scatter = px.scatter(
                        df_temp_scatter.dropna(subset=["_X_Axis_", "_Y_Axis_"]),
                        x="_X_Axis_",
                        y="_Y_Axis_",
                        title=f"Relation entre '{FEATURE_DESCRIPTIONS.get(feature_x_tech_name, feature_x_tech_name)}' et '{FEATURE_DESCRIPTIONS.get(feature_y_tech_name, feature_y_tech_name)}'",
                        opacity=0.6,
                        hover_data={feature_x_tech_name: True, feature_y_tech_name: True},
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    if client_x_val is not None and client_y_val is not None:
                        fig_scatter.add_trace(go.Scatter(
                            x=[client_x_val],
                            y=[client_y_val],
                            mode='markers',
                            marker=dict(color='red', size=12, symbol='star'),
                            name='Client Actuel',
                            hovertemplate=f"Client X: {client_x_val}<br>Client Y: {client_y_val}"
                        ))
                    fig_scatter.update_layout(height=400,
                                              xaxis_title=FEATURE_DESCRIPTIONS.get(
                                                  feature_x_tech_name, feature_x_tech_name),
                                              yaxis_title=FEATURE_DESCRIPTIONS.get(
                                                  feature_y_tech_name, feature_y_tech_name))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning(
                        "Impossible de générer le graphique bi-varié car une ou plusieurs caractéristiques n'ont pas pu être traitées.")
            else:
                st.warning(
                    "Caractéristiques non trouvées pour l'analyse bi-variée dans les données de référence.")

        st.markdown("---")
        st.subheader("🔍 Autres Graphiques Pertinents")

        if 'AMT_CREDIT' in df_ref.columns and 'NAME_EDUCATION_TYPE' in df_ref.columns:
            df_temp_box = df_ref.copy()

            df_temp_box['NAME_EDUCATION_TYPE'] = df_temp_box['NAME_EDUCATION_TYPE'].astype(str)

            fig_box = px.box(df_temp_box, x="NAME_EDUCATION_TYPE", y="AMT_CREDIT",
                             title="Montant du crédit par niveau d'éducation",
                             labels={"NAME_EDUCATION_TYPE": FEATURE_DESCRIPTIONS.get(
                                 "NAME_EDUCATION_TYPE", "Niveau d'éducation"),
                                     "AMT_CREDIT": FEATURE_DESCRIPTIONS.get("AMT_CREDIT",
                                                                            "Montant du crédit")},
                             color="NAME_EDUCATION_TYPE",
                             color_discrete_map={
                                 "Higher education": "blue",
                                 "Secondary / secondary special": "green",
                                 "Incomplete higher": "orange",
                                 "Lower secondary": "red",
                                 "Academic degree": "purple",
                                 "nan": "gray"
                             })
            client_edu = client_data_for_api.get('NAME_EDUCATION_TYPE')
            client_credit = client_data_for_api.get('AMT_CREDIT')
            if client_edu and client_credit is not None:
                fig_box.add_trace(go.Scatter(
                    x=[client_edu],
                    y=[client_credit],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Client Actuel',
                    hovertemplate=f"Client Éducation: {client_edu}<br>Client Crédit: {client_credit}"
                ))
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

# --- Critères d'accessibilité WCAG (implémentation partielle) ---
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