# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:44:56 2025

@author: Eliezer
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

from fpdf import FPDF
import base64

# Génération des données réalistes
maladies = [
    "Paludisme", "Tuberculose", "VIH/SIDA", "Diabète", "Hypertension artérielle",
    "Cardiopathies", "AVC", "Cancer du sein", "Cancer du poumon", "Hépatite B et C",
    "Grippe", "Pneumonie", "Asthme", "Maladie rénale chronique", "Dépression",
    "Troubles anxieux", "Obésité", "Maladie d'Alzheimer", "Arthrite", "COVID-19"
]

# Dictionnaire des maladies avec leur spécialité correspondante
specialites = {
    "Paludisme": "Médecine générale",
    "Tuberculose": "Pneumologie",
    "VIH/SIDA": "Infectiologie",
    "Diabète": "Endocrinologie",
    "Hypertension artérielle": "Cardiologie",
    "Cardiopathies": "Cardiologie",
    "AVC": "Neurologie",
    "Cancer du sein": "Oncologie médicale",
    "Cancer du poumon": "Oncologie médicale",
    "Hépatite B et C": "Hépatologie",
    "Grippe": "Médecine générale",
    "Pneumonie": "Pneumologie",
    "Asthme": "Pneumologie",
    "Maladie rénale chronique": "Néphrologie",
    "Dépression": "Psychiatrie",
    "Troubles anxieux": "Psychiatrie",
    "Obésité": "Endocrinologie",
    "Maladie d'Alzheimer": "Gériatrie",
    "Arthrite": "Rhumatologie",
    "COVID-19": "Médecine générale / Infectiologie"
}

symptoms_dict = {
    "Paludisme": ["Fièvre", "Sueurs nocturnes", "Maux de tête", "Fatigue intense"],
    "Tuberculose": ["Toux persistante", "Perte de poids", "Sueurs nocturnes", "Fatigue intense"],
    "VIH/SIDA": ["Fièvre", "Perte de poids", "Sueurs nocturnes", "Fatigue intense"],
    "Diabète": ["Fatigue intense", "Soif excessive", "Urination fréquente", "Vision trouble"],
    "Hypertension artérielle": ["Maux de tête", "Vertiges", "Douleurs thoraciques", "Essoufflement"],
    "Cardiopathies": ["Douleurs thoraciques", "Essoufflement", "Palpitations", "Fatigue intense"],
    "AVC": ["Confusion", "Engourdissement", "Troubles de la parole", "Faiblesse musculaire"],
    "Cancer du sein": ["Masse mammaire", "Douleurs mammaires", "Changement de la peau", "Écoulement mammaire"],
    "Cancer du poumon": ["Toux persistante", "Douleurs thoraciques", "Essoufflement", "Fatigue intense"],
    "Hépatite B et C": ["Fatigue intense", "Jaunisse", "Douleurs abdominales", "Nausées"],
    "Grippe": ["Fièvre", "Toux", "Maux de tête", "Douleurs musculaires"],
    "Pneumonie": ["Fièvre", "Toux", "Essoufflement", "Douleurs thoraciques"],
    "Asthme": ["Essoufflement", "Toux nocturne", "Oppression thoracique", "Sifflements"],
    "Maladie rénale chronique": ["Fatigue intense", "Urination fréquente", "Douleurs lombaires", "Nausées"],
    "Dépression": ["Fatigue intense", "Perte d'intérêt", "Troubles du sommeil", "Humeur triste"],
    "Troubles anxieux": ["Palpitations", "Transpiration excessive", "Agitation", "Tremblements"],
    "Obésité": ["Prise de poids", "Fatigue intense", "Douleurs articulaires", "Essoufflement"],
    "Maladie d'Alzheimer": ["Troubles de mémoire", "Confusion", "Désorientation", "Difficulté à parler"],
    "Arthrite": ["Douleurs articulaires", "Raideur matinale", "Gonflement des articulations", "Fatigue intense"],
    "COVID-19": ["Fièvre", "Toux sèche", "Essoufflement", "Perte de goût ou d'odorat"]
}

# Base de médecins affiliés
medecins_affilies = {
    "Médecine générale": [
        {"nom": "Dr. Kouadio", "hopital": "Clinique Sainte Marie", "contact": "+225 07 01 23 45"},
        {"nom": "Dr. Yapi", "hopital": "Clinique Sainte Marie", "contact": "+225 07 02 02 45"},
        {"nom": "Dr. Konan", "hopital": "CHU de Yopougon", "contact": "+225 07 03 90 45"},
    ],
    "Pneumologie": [
        {"nom": "Dr. Diarra", "hopital": "CHU de Treichville - Service Pneumologie", "contact": "+225 05 66 77 88"},
    ],
    "Infectiologie": [
        {"nom": "Dr. Koffi", "hopital": "Institut Pasteur de Côte d'Ivoire", "contact": "+225 01 22 33 44"},
    ],
    "Endocrinologie": [
        {"nom": "Dr. Traoré", "hopital": "CHU de Cocody - Service Endocrinologie", "contact": "+225 05 55 66 77"},
    ],
    "Cardiologie": [
        {"nom": "Dr. Yao", "hopital": "Institut de Cardiologie d'Abidjan", "contact": "+225 01 99 88 77"},
    ],
    "Neurologie": [
        {"nom": "Dr. N'Guessan", "hopital": "CHU de Yopougon - Service Neurologie", "contact": "+225 07 88 99 11"},
    ],
    "Oncologie médicale": [
        {"nom": "Dr. Kouakou", "hopital": "Institut National d’Oncologie", "contact": "+225 01 77 55 44"},
    ],
    "Hépatologie": [
        {"nom": "Dr. Aka", "hopital": "CHU de Treichville - Service Gastro-Hépatologie", "contact": "+225 05 33 44 55"},
    ],
    "Néphrologie": [
        {"nom": "Dr. Fofana", "hopital": "CHU de Cocody - Service Néphrologie", "contact": "+225 07 22 11 33"},
    ],
    "Psychiatrie": [
        {"nom": "Dr. Amani", "hopital": "Centre Psychiatrique Bingerville", "contact": "+225 01 44 55 66"},
    ],
    "Gériatrie": [
        {"nom": "Dr. Adjé", "hopital": "CHU de Treichville - Service Gériatrie", "contact": "+225 05 99 77 88"},
    ],
    "Rhumatologie": [
        {"nom": "Dr. N'Dri", "hopital": "CHU de Yopougon - Service Rhumatologie", "contact": "+225 01 88 22 44"},
    ],
    "Médecine générale / Infectiologie": [
        {"nom": "Dr. Tano", "hopital": "CHU de Cocody - Unité des Maladies Infectieuses", "contact": "+225 07 66 55 44"},
    ],
}



# Liste statique des symptômes
symptoms = [
    'Fièvre', 'Palpitations', 'Douleurs thoraciques', 'Troubles de la parole', 'Douleurs mammaires', 
    'Douleurs lombaires', 'Tremblements', 'Perte de poids', 'Maux de tête', 'Difficulté à parler', 
    'Masse mammaire', 'Essoufflement', 'Fatigue intense', 'Agitation', 'Troubles de mémoire', 'Engourdissement', 
    'Troubles du sommeil', 'Vertiges', 'Oppression thoracique', 'Douleurs musculaires', 'Gonflement des articulations', 
    'Changement de la peau', 'Raideur matinale', 'Toux persistante', 'Vision trouble', 'Transpiration excessive', 
    'Douleurs articulaires', 'Jaunisse', 'Faiblesse musculaire', 'Écoulement mammaire', 'Urination fréquente', 
    'Soif excessive', 'Toux nocturne', 'Toux sèche', 'Prise de poids', 'Désorientation', 'Sueurs nocturnes', 
    'Confusion', 'Douleurs abdominales', 'Toux', 'Perte de goût ou d\'odorat', 'Humeur triste', 'Nausées', 
    'Perte d\'intérêt', 'Sifflements'
]



def export_pdf(name, surname, age, height, weight, sex, predicted_disease, confidence, bmi, bmr, specialite, responses, medecins):
    pdf = FPDF()
    pdf.add_page()

    # En-tête style ordonnance
    pdf.set_fill_color(200, 220, 255)  # Bleu clair
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 12, "Rapport Médical - Diagnostic AI", ln=True, align="C", fill=True)
    pdf.ln(5)
    pdf.set_draw_color(0, 0, 0)
    pdf.line(10, 25, 200, 25)  # ligne de séparation
    pdf.ln(10)

    # Infos personnelles encadrées
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Informations Personnelles", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Nom : {surname} {name}", ln=True)
    pdf.cell(0, 8, f"Âge : {age} ans", ln=True)
    pdf.cell(0, 8, f"Sexe : {sex}", ln=True)
    pdf.cell(0, 8, f"Taille : {height} cm", ln=True)
    pdf.cell(0, 8, f"Poids : {weight} kg", ln=True)
    pdf.ln(5)

    # Résultat du diagnostic
    pdf.set_fill_color(255, 230, 230)  # rose clair
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Résultat du diagnostic", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Maladie probable : {predicted_disease}", ln=True)
    pdf.cell(0, 8, f"Précision du modèle : {confidence:.2f}%", ln=True)
    pdf.cell(0, 8, f"Spécialité recommandée : {specialite}", ln=True)
    pdf.ln(5)

    # Indicateurs santé
    pdf.set_fill_color(230, 255, 230)  # vert clair
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Indicateurs de santé", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"IMC : {bmi:.2f}", ln=True)
    pdf.cell(0, 8, f"BMR : {bmr:.2f} kcal/jour", ln=True)
    pdf.ln(5)

    # Symptômes
    pdf.set_fill_color(255, 255, 200)  # jaune clair
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Symptômes déclarés", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", '', 12)
    for sympt, val in responses.items():
        if val == 1:
            pdf.cell(0, 8, f"- {sympt}", ln=True)
    pdf.ln(5)

    # Médecins affiliés (tableau)
    if medecins:
        pdf.set_fill_color(200, 220, 255)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Médecins recommandés", ln=True, fill=True)
        pdf.ln(5)

        # En-tête du tableau
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(60, 10, "Nom", 1, 0, 'C', fill=True)
        pdf.cell(70, 10, "Hôpital", 1, 0, 'C', fill=True)
        pdf.cell(60, 10, "Contact", 1, 1, 'C', fill=True)

        # Contenu du tableau
        pdf.set_font("Arial", '', 12)
        for medecin in medecins:
            pdf.cell(60, 10, medecin['nom'], 1, 0)
            pdf.cell(70, 10, medecin['hopital'], 1, 0)
            pdf.cell(60, 10, medecin['contact'], 1, 1)
        pdf.ln(5)
    else:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 10, "Aucun médecin enregistré pour cette spécialité.", ln=True)

    # Footer simple
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Rapport généré automatiquement - AI Medical Diagnostic", ln=True, align='C')

    # Sauvegarde
    file_path = "rapport_medical.pdf"
    pdf.output(file_path)
    return file_path








# Interface utilisateur
st.title("Système de Diagnostic Médical AI")

# Étape 1 : Saisie des informations utilisateur
st.header("Informations Personnelles")
name = st.text_input("Nom")
surname = st.text_input("Prénom")
age = st.number_input("Âge", min_value=0, max_value=120, step=1)
height = st.number_input("Taille (cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Poids (kg)", min_value=10, max_value=300, step=1)
sex = st.radio("Sexe", ("Homme", "Femme"))

if st.button("Suivant"):
    st.session_state["page"] = "questionnaire"

# Étape 2 : Questionnaire
if "page" in st.session_state and st.session_state["page"] == "questionnaire":
    st.header("Répondez aux questions suivantes")
    responses = {}

    for symptom in symptoms:
        # Affichage du symptôme en gras et en plus grand
        st.markdown(f"<h3 style='font-weight: bold; font-size:24px;'>{symptom} ?</h3>", unsafe_allow_html=True)
        
        # Utilisation d'une clé unique pour chaque bouton radio
        answer = st.radio("", ("Oui", "Non"), index=1, key=symptom)
        
        # Stocker la réponse sous forme binaire
        responses[symptom] = 1 if answer == "Oui" else 0

    # Bouton pour passer à la phase de diagnostic
    if st.button("Obtenir un diagnostic"):
        st.session_state["responses"] = responses
        st.session_state["page"] = "diagnostic"


# Étape 3 : Diagnostic final
if "page" in st.session_state and st.session_state["page"] == "diagnostic":
    st.header("Résultat du Diagnostic")
    #loaded_model = joblib.load(r"C:\Users\Eliezer\Projets_methode_avancee\RandomForest_model.pkl")
    loaded_model = joblib.load(r"C:\Users\tanoh\Projets_methode_avancee\RandomForest_model.pkl")
    user_input = np.array(list(st.session_state["responses"].values())).reshape(1, -1)
    predicted_disease = loaded_model.predict(user_input)[0]
    confidence = max(loaded_model.predict_proba(user_input)[0]) * 100
    
    st.write(f"### Diagnostic probable : {predicted_disease}")
    st.write(f"### Précision du modèle : {confidence:.2f}%")
    st.write("Vos réponses ont été enregistrées pour améliorer le modèle.")
    
    # Informations de santé basées sur la taille, le poids, l'âge et le sexe
    st.subheader("Informations de santé supplémentaires")
    
    # Calcul de l'IMC
    bmi = weight / (height / 100) ** 2
    st.write(f"### Indice de Masse Corporelle (IMC) : {bmi:.2f}")
    
    # Catégorie de l'IMC
    if bmi < 18.5:
        st.write("Vous êtes en sous-poids.")
    elif 18.5 <= bmi < 24.9:
        st.write("Vous avez un poids normal.")
    elif 25 <= bmi < 29.9:
        st.write("Vous êtes en surpoids.")
    else:
        st.write("Vous êtes en obésité.")

    # Risques liés à l'âge, au sexe et à l'IMC
    if age > 50:
        st.write("À partir de 50 ans, les risques de maladies cardiovasculaires et de diabète augmentent.")
    if sex == "Femme" and age > 50:
        st.write("Les femmes après la ménopause sont à risque accru d'ostéoporose.")
    
    # Recommandations générales
    st.write("Il est recommandé de maintenir une activité physique régulière et une alimentation équilibrée.")
    
    # Indicateurs anthropométriques
    st.subheader("Indicateurs Anthropométriques")
    st.write(f"### Taille : {height} cm")
    st.write(f"### Poids : {weight} kg")
    
    # Risques de maladies métaboliques
    if bmi >= 25:
        st.write("Risque accru de diabète de type 2, d'hypertension et de maladies cardiaques.")
    
    # Métabolisme et besoins énergétiques
    bmr = 10 * weight + 6.25 * height - 5 * age
    if sex == "Homme":
        bmr += 5
    else:
        bmr -= 161
    st.write(f"### Besoins énergétiques (BMR) : {bmr:.2f} kcal/jour")
    
    # Développement et vieillissement
    if age <= 18:
        st.write("Attention à la croissance et au développement, assurez-vous de consommer des nutriments essentiels.")
    if age >= 60:
        st.write("Risque accru de perte musculaire et de problèmes osseux, veillez à une bonne nutrition et à l'exercice physique.")
    
    # Spécialité médicale associée à la maladie prédite
    st.subheader("Spécialité recommandée")
    st.write(f"### Vous pouvez consulter un spécialiste en {specialites.get(predicted_disease, 'Médecine générale')} pour plus de détails.")
    
    specialite = specialites.get(predicted_disease, "Médecine générale")
    
    # Recommandation médecin affilié

    if specialite in medecins_affilies:
        st.subheader("Médecins affiliés recommandés")
    
        # Préparer les données pour le tableau
        data = []
        for medecin in medecins_affilies[specialite]:
            whatsapp_link = f"https://wa.me/{medecin['contact'].replace(' ', '').replace('+', '')}"
            data.append({
                "Nom": medecin['nom'],
                "Hôpital": medecin['hopital'],
                "Contact": medecin['contact'],
                "WhatsApp": f"[Contacter]({whatsapp_link})"
            })
    
        # Convertir en DataFrame
        df_medecins = pd.DataFrame(data)
    
        # Afficher le tableau avec markdown cliquable pour WhatsApp
        st.markdown(
            df_medecins.to_markdown(index=False),
            unsafe_allow_html=True
        )
    
    else:
        st.info("Aucun médecin affilié enregistré pour cette spécialité pour le moment.")

    # Dans la section DIAGNOSTIC
    if st.button("Exporter en PDF"):
        file_path = export_pdf(
            name, surname, age, height, weight, sex,
            predicted_disease, confidence, bmi, bmr,
            specialites.get(predicted_disease, "Médecine générale"),
            st.session_state["responses"],
            medecins_affilies.get(specialite, [])  # récupération des médecins de cette spécialité
        )

        
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="rapport_medical.pdf">Télécharger le rapport PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

