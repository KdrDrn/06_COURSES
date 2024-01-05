import streamlit as st

# Function to get or create session state
def get_session_state():
    session_state = st.session_state
    if not hasattr(session_state, "personal_dict"):
        session_state.personal_dict = {}
    return session_state

def add_word():
    session_state = get_session_state()
    word = st.text_input("Enter the word:")
    definition = st.text_input("Enter the definition:")
    if st.button("Add Word"):
        if word and definition:
            session_state.personal_dict[word] = definition
            st.success(f"Word '{word}' added to the dictionary.")
        else:
            st.warning("Please enter both word and definition.")

def get_definition():
    session_state = get_session_state()
    word = st.text_input("Enter the word to get the definition:")
    if st.button("Get Definition"):
        if word:
            definition = session_state.personal_dict.get(word, f"Sorry, '{word}' not found in the dictionary.")
            st.info(definition)
        else:
            st.warning("Please enter a word.")

def list_words_alphabetically():
    session_state = get_session_state()
    sorted_words = sorted(session_state.personal_dict.items())
    text = "\n".join([f"{word}: {definition}" for word, definition in sorted_words])
    st.text_area("Words Alphabetically", text)

# Streamlit app layout
st.title("Personalized Dictionary")

add_word()
get_definition()
list_words_alphabetically()
