import streamlit as st
import joblib
import re
import string
import random
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model and vectorizer
rf_model = joblib.load('stress_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text preprocessing function
def clean(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    stemmer = PorterStemmer()
    stopword = set(stopwords.words('english'))
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stopword])
    return text

# Motivational Quotes
quotes = [
    "You are stronger than you think! 💪",
    "Take a deep breath, you're doing great! 😊",
    "Your mind is powerful. Train it to see the good in everything. 🌿",
    "Every storm passes. Keep pushing forward! ⛅",
    "Small steps every day lead to big changes. 🌟"
]

# Initialize session state for tracking stress levels
if 'stress_history' not in st.session_state:
    st.session_state['stress_history'] = []

# Streamlit App
st.set_page_config(page_title="Stress Detection App", page_icon=":relieved:", layout="centered")
st.title("🌟 Stress Detection App 🌟")

st.markdown("""
    This app analyzes your text to predict stress levels and provides personalized stress management tips!
    
    ### How does it work?
    - The app uses a machine learning model trained on stress-related text.
    - It preprocesses your input and predicts stress levels.
    - Offers **stress relief tools**, **motivational quotes**, and **stress tracking**.
    """)

# User input
user_input = st.text_area("Type your text here:", height=200)

# Customizable Stress Relief Toolkit
st.subheader("🎒 Customize Your Stress Relief Toolkit")
stress_relief_options = ["Deep Breathing 🧘", "Listen to Music 🎵", "Go for a Walk 🚶", "Meditation 🌿", "Talk to a Friend ☎️", "Journaling ✍️"]
selected_relief_methods = st.multiselect("Choose your favorite stress relief methods:", stress_relief_options)

# Prediction Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing your text..."):
            cleaned_text = clean(user_input)
            vectorized_text = vectorizer.transform([cleaned_text]).toarray()
            prediction = rf_model.predict(vectorized_text)
            stress_score = random.randint(40, 100) if prediction[0] == "Stress" else random.randint(0, 39)  # Simulated score

            # Display result
            result_text = "😟 Stress Detected" if prediction[0] == "Stress" else "😊 No Stress Detected"
            st.subheader(f"📊 Stress Score: {stress_score}/100")
            st.success(f"**{result_text}**")

            # Store stress history
            st.session_state['stress_history'].append((len(st.session_state['stress_history']) + 1, stress_score))

            # Suggestions
            if prediction[0] == "Stress":
                st.warning("It looks like you're experiencing stress. Try these:")
                for method in selected_relief_methods:
                    st.write(f"✅ {method}")
                st.markdown(f"**💡 Tip:** {random.choice(quotes)}")
            else:
                st.info("You're in a good state! Keep up the positive energy! 💙")

# Stress Level Graph
if len(st.session_state['stress_history']) > 1:
    st.subheader("📈 Your Stress Level Over Time")
    df = pd.DataFrame(st.session_state['stress_history'], columns=["Entry", "Stress Score"])
    fig, ax = plt.subplots()
    ax.plot(df["Entry"], df["Stress Score"], marker='o', linestyle='-', color='red')
    ax.set_xlabel("Entries")
    ax.set_ylabel("Stress Score")
    ax.set_title("Stress Level Trend")
    st.pyplot(fig)

# Footer
st.markdown("""
    ---
    _Stay mindful and take care of your mental well-being!_ 🌿
    
    Contact us at: **support@stressdetect.com**
    """)
