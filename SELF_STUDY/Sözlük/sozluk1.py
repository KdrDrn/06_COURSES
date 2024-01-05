import streamlit as st
import requests

# Function to get or create session state
def get_session_state():
    session_state = st.session_state
    if not hasattr(session_state, "personal_dict"):
        session_state.personal_dict = {}
    return session_state

# Function to get word definition from WordsAPI
def get_word_definition(word):
    url = f'https://wordsapiv1.p.rapidapi.com/words/{word}'
    headers = {
        'X-RapidAPI-Host': 'wordsapiv1.p.rapidapi.com',
        'X-RapidAPI-Key': 'your_wordsapi_key'  # Replace with your actual API key
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'results' in data and 'definition' in data['results']:
            return data['results']['definition']
    return None

# Rest of your Streamlit app code...

def get_definition():
    session_state = get_session_state()

    language_options = ["Select Language"] + list(session_state.personal_dict.keys())
    selected_language = st.selectbox("Select Language:", language_options, key="get_language")

    word = st.text_input(f"Enter the word to get the definition in {selected_language}:")

    if st.button("Get Definition"):
        if selected_language in session_state.personal_dict:
            if word and word in session_state.personal_dict[selected_language]:
                # Fetch definition from WordsAPI
                definition = get_word_definition(word)
                if definition:
                    st.info(f"Language: {selected_language}\nDefinition: {definition}")
                else:
                    st.warning(f"Definition not found for '{word}' in the online dictionary.")
            else:
                st.warning(f"Word '{word}' not found in the dictionary for '{selected_language}' language.")
        else:
            st.info(f"No words found in the dictionary for '{selected_language}' language.")



# Function to get or create session state
def get_session_state():
    session_state = st.session_state
    if not hasattr(session_state, "personal_dict"):
        session_state.personal_dict = {}
    return session_state

def add_word():
    session_state = get_session_state()

    # Move the "Enter a new language" input to the sidebar
    new_language = st.sidebar.text_input("Enter a new language:")
    if new_language and new_language not in session_state.personal_dict:
        session_state.personal_dict[new_language] = {}

    language_options = ["Select Language"] + list(session_state.personal_dict.keys())
    selected_language = st.selectbox("Select Language:", language_options, key="add_language")

    word = st.text_input("Enter the word:")
    definition = st.text_input("Enter the definition:")
    sentence = st.text_input("Enter a sentence (optional):")

    if st.button("Add Word"):
        if word and definition and selected_language:
            if selected_language == "Select Language":
                st.warning("Please select a language.")
            else:
                word_info = {"Definition": definition, "Sentence": sentence}
                session_state.personal_dict[selected_language][word] = word_info
                st.success(f"Word '{word}' added to the dictionary in '{selected_language}' language.")
        else:
            st.warning("Please enter both word, definition, and select a language.")

def get_definition():
    session_state = get_session_state()

    language_options = ["Select Language"] + list(session_state.personal_dict.keys())
    selected_language = st.selectbox("Select Language:", language_options, key="get_language")

    word = st.text_input(f"Enter the word to get the definition in {selected_language}:")

    if st.button("Get Definition"):
        if selected_language in session_state.personal_dict:
            if word and word in session_state.personal_dict[selected_language]:
                word_info = session_state.personal_dict[selected_language][word]
                st.info(f"Language: {selected_language}\nDefinition: {word_info['Definition']}{' - Sentence: '+word_info['Sentence'] if word_info.get('Sentence') else ''}")
            else:
                st.warning(f"Word '{word}' not found in the dictionary for '{selected_language}' language.")
        else:
            st.info(f"No words found in the dictionary for '{selected_language}' language.")

def list_words_alphabetically():
    session_state = get_session_state()

    language_options = ["Select Language"] + list(session_state.personal_dict.keys())
    selected_language = st.selectbox("Select Language:", language_options, key="list_language")

    if selected_language != "Select Language":
        if selected_language in session_state.personal_dict:
            sorted_words = sorted(session_state.personal_dict[selected_language].items())
            text = "\n".join([f"{word}: {info['Definition']}{' - Sentence: '+info['Sentence'] if info.get('Sentence') else ''}" for word, info in sorted_words])
            st.text_area(f"Words Alphabetically in {selected_language}", text)
        else:
            st.info(f"No words found in the dictionary for '{selected_language}' language.")

def show_letters_and_words():
    session_state = get_session_state()

    language_options = ["Select Language"] + list(session_state.personal_dict.keys())
    selected_language = st.selectbox("Select Language:", language_options, key="letter_language")

    if selected_language != "Select Language":
        st.subheader(f"Words by Letter in {selected_language}")
        letters_dict = {}

        if selected_language in session_state.personal_dict:
            for word in session_state.personal_dict[selected_language]:
                if word:
                    first_letter = word[0].upper()
                    if first_letter not in letters_dict:
                        letters_dict[first_letter] = []
                    letters_dict[first_letter].append(word)

            for letter, words in sorted(letters_dict.items()):
                st.write(f"**{letter}**: {', '.join(words)}")

# Streamlit app layout
st.title("Personalized Dictionary")

add_word()
get_definition()
list_words_alphabetically()
show_letters_and_words()
