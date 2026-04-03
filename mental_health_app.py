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
# CHATBOT
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
st.subheader("💬 Your Support Companion")

user_input = st.text_input("Talk to me... I'm here for you 🤍")

# -----------------------------
# CHATBOT FUNCTION (ONLY LOGIC)
# -----------------------------
def chatbot_reply(user_text):
    text = user_text.lower()

    # Get previous context
    history = st.session_state.chat_history[-6:]
    previous_msgs = [msg for speaker, msg in history if speaker == "You"]
    context = " ".join(previous_msgs).lower()

    name = "Paapu"

    if "not okay" in text or "sad" in text:
        return f"{name}… 🤍 I can feel something is heavy… I'm here with you. Tell me what happened."

    elif "stress" in text:
        return f"Hey {name}… 😔 that sounds really exhausting… what’s stressing you?"

    elif "study" in text or "exam" in text:
        return f"Ahh {name}… studies can be really overwhelming 😣 Are exams coming?"

    elif "stress" in context:
        return f"I see {name}… this is what's been stressing you 😔 tell me more…"

    else:
        return f"I'm here for you {name} 💙 tell me anything…"


# -----------------------------
# SEND BUTTON (OUTSIDE FUNCTION)
# -----------------------------
if st.button("Send 💬"):
    if user_input.strip() != "":
        response = chatbot_reply(user_input)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))


# -----------------------------
# DISPLAY CHAT
# -----------------------------
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.write(f"🧍‍♀️ **You:** {msg}")
    else:
        st.write(f"🤖 **Companion:** {msg}")
