import streamlit as st
from google_trans_new import google_translator

# Function to get word translation using Google Translate
def get_word_translation(word, dest_language):
    translator = google_translator()
    translation = translator.translate(word, lang_tgt=dest_language)
    return translation

# Streamlit app layout
st.title("Multilingual Personalized Dictionary")

# Define the order of languages for translation
translation_order = ["tr", "en", "bs", "de", "es", "ru"]

# Sidebar for language selection
selected_language = st.sidebar.selectbox("Select Language:", ["tr", "en", "bs", "de", "es", "ru"])

# Sidebar for entering word
word = st.sidebar.text_input("Enter a word:")

# Main area for displaying translations
if st.sidebar.button("Get Translations"):
    if word and selected_language:
        st.subheader(f"Word Translations for '{word}' in {selected_language.capitalize()}:")
        
        for dest_language in translation_order:
            if dest_language != selected_language:
                translation = get_word_translation(word, dest_language)
                
                if translation:
                    st.markdown(f"**{dest_language.capitalize()} Translation:** {translation}")
                else:
                    st.warning(f"Translation not found for '{word}' in {dest_language}.")
    else:
        st.warning("Please enter a word and select a language.")
