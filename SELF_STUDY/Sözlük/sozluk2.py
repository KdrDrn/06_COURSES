import streamlit as st
from translate import Translator

# Function to get word translation in all selected languages
def get_translations_in_selected_languages(word):
    translation_order = ["tr", "en", "bs", "de", "es", "ru"]
    translations = {}

    for dest_language in translation_order:
        if dest_language != "en":  # Exclude English, as we'll display the English word
            translator = Translator(to_lang=dest_language)
            translation = translator.translate(word)
            translations[dest_language] = translation

    return translations

# Streamlit app layout with HTML and CSS
st.title("Multilingual Personalized Dictionary")

# Add custom CSS styles
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 800px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        input {
            padding: 10px;
            width: 80%;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for entering English word
english_word = st.sidebar.text_input("Enter an English word:", key="english_word")

# Main area for displaying translations
if st.sidebar.button("Get Translations"):
    if english_word:
        translations = get_translations_in_selected_languages(english_word)
        
        st.markdown(f"<h2>Word Translations for '{english_word}':</h2>", unsafe_allow_html=True)
        
        st.markdown(f"<p><strong>English Word:</strong> {english_word}</p>", unsafe_allow_html=True)
        
        for dest_language, translation in translations.items():
            st.markdown(f"<p><strong>{dest_language.capitalize()} Translation:</strong> {translation}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter an English word.")
