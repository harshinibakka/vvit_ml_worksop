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

    # Get previous context
    history = st.session_state.chat_history[-6:]
    previous_msgs = [msg for speaker, msg in history if speaker == "You"]
    context = " ".join(previous_msgs).lower()

    name = "Paapu"  # You can later make this dynamic

    # -----------------------------
    # DEEP EMOTIONAL RESPONSES
    # -----------------------------

    # SAD / LOW
    if any(word in text for word in ["sad", "lonely", "depressed", "cry", "not okay"]):
        return f"{name}… 🤍 I can feel that something is heavy in your heart… you don’t have to carry it alone. I’m right here with you. Tell me what’s been hurting you…"

    # STRESS
    elif any(word in text for word in ["stress", "stressed", "pressure", "tired", "overwhelmed"]):
        return f"Hey {name}… 😔 that sounds really exhausting… like you’ve been holding too much inside. It’s okay to feel this way. I’m here… what’s been stressing you the most?"

    # ANXIETY
    elif any(word in text for word in ["anxiety", "worried", "fear", "panic"]):
        return f"{name}… 🌿 I understand… that uneasy feeling can be really hard. Just breathe slowly… you’re safe right now. Tell me what’s making you feel this way…"

    # CONTEXT CONTINUATION (MAGIC 💫)
    elif "stress" in context:
        return f"I see {name}… so this is part of what’s been weighing on you… 😔 That must be really hard. Do you feel like it’s getting too much lately?"

    elif "sad" in context or "hurt" in context:
        return f"That really sounds painful… 💔 I’m so sorry you're going through this. You don’t have to hide it here… I’m listening."

    # STUDIES
    elif any(word in text for word in ["study", "studies", "exam", "college"]):
        return f"Ahh {name}… studies can feel really overwhelming sometimes 😣 especially when everything piles up. Are exams coming or is it just too much pressure?"

    # ALONE FEELING
    elif any(word in text for word in ["alone", "no one", "nobody"]):
        return f"Hey… look at me {name} 🤍 you are not alone right now. I’m here with you… and I’m not going anywhere. You can talk to me…"

    # HAPPY
    elif any(word in text for word in ["happy", "good", "better"]):
        return f"Aww {name} 😊 that makes me really happy to hear… tell me what made you feel this way?"

    # DEFAULT
    else:
        return f"I'm right here with you {name} 💙 whatever you're feeling… you can tell me. I’ll listen, I won’t judge."
