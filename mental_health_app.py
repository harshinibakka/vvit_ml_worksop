import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("mental_health_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Mental Health Prediction", layout="centered")

st.title("🧠 Mental Health Risk Prediction")
st.write("Fill in the details to check mental health risk")

# -----------------------------
# INPUT FIELDS
# -----------------------------
age = st.slider("Age", 18, 60, 25)

gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History", ["Yes", "No"])
work_interfere = st.selectbox("Work Interference", ["Never", "Rarely", "Sometimes", "Often"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
benefits = st.selectbox("Company Benefits", ["Yes", "No"])
care_options = st.selectbox("Care Options Available", ["Yes", "No"])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):

    # HIGH RISK (RULE)
    if family_history == "Yes" and work_interfere in ["Often", "Sometimes"]:
        st.error("🔴 High Risk of Mental Health Issues")
        st.write("👉 Suggestions:")
        st.write("- Improve work-life balance")
        st.write("- Seek professional help")
        st.write("- Talk to HR / support system")

    # LOW RISK (RULE)
    elif work_interfere == "Never" and benefits == "Yes" and care_options == "Yes":
        st.success("🟢 Low Risk of Mental Health Issues")
        st.write("👉 Keep maintaining a healthy lifestyle 😊")

    # ML LOGIC
    else:
        input_dict = {
            "Age": age,
            "Gender": gender,
            "family_history": family_history,
            "work_interfere": work_interfere,
            "remote_work": remote_work,
            "benefits": benefits,
            "care_options": care_options
        }

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=columns, fill_value=0)

        proba = model.predict_proba(input_df)[0]

        if proba[1] > 0.6:
            st.error("🔴 High Risk of Mental Health Issues")
            st.write("👉 Suggestions:")
            st.write("- Improve work-life balance")
            st.write("- Seek professional help")
            st.write("- Talk to HR / support system")

        elif proba[1] > 0.4:
            st.warning("🟡 Medium Risk of Mental Health Issues")
            st.write("👉 Suggestions:")
            st.write("- Take short breaks")
            st.write("- Maintain work-life balance")
            st.write("- Talk to someone you trust")

        else:
            st.success("🟢 Low Risk of Mental Health Issues")
            st.write("👉 Keep maintaining a healthy lifestyle 😊")

# -----------------------------
# CHATBOT (OUTSIDE PREDICTION)
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
st.subheader("💬 Your Support Companion")

user_input = st.text_input("Talk to me... I'm here for you 🤍")

def chatbot_reply(user_text):
    text = user_text.lower()

    if any(word in text for word in ["sad", "lonely", "depressed", "cry"]):
        return "I'm really sorry you're feeling this way 🤍 I'm here with you. Do you want to share what happened?"

    elif any(word in text for word in ["stress", "tired", "pressure", "overwhelmed"]):
        return "That sounds really heavy… 😔 You're handling a lot. Tell me what's stressing you."

    elif any(word in text for word in ["anxiety", "worried", "fear", "panic"]):
        return "I understand… take a slow breath 🌿 You're safe. What’s making you feel this way?"

    elif any(word in text for word in ["happy", "good", "better"]):
        return "That’s nice to hear 😊 I’m really glad. Want to share more?"

    elif any(word in text for word in ["alone", "no one", "nobody"]):
        return "Hey… you're not alone 🤍 I'm here with you."

    else:
        return "I'm here for you 💙 Tell me anything…"

# Store chat
if user_input:
    response = chatbot_reply(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.write(f"🧍‍♀️ **You:** {msg}")
    else:
        st.write(f"🤖 **Companion:** {msg}")
