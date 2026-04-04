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
# FINAL SMART CHATBOT
# -----------------------------

# Initialize memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_state" not in st.session_state:
    st.session_state.user_state = {
        "topic": None,
        "emotion": None,
        "last_question": None
    }

st.markdown("---")
st.subheader("💬 Your Support Companion")

# INPUT BOX
user_input = st.text_input("Talk to me... I'm here for you 🤍", key="input_box")

# -----------------------------
# CHATBOT FUNCTION
# -----------------------------
def chatbot_reply(user_text):
    text = user_text.lower()

    # Memory
    history = st.session_state.chat_history[-6:]
    context = " ".join([msg for speaker, msg in history if speaker == "You"]).lower()

    state = st.session_state.user_state

    # Detect emotion
    if any(word in text for word in ["stress", "stressed"]):
        state["emotion"] = "stress"

    if any(word in text for word in ["exam", "study"]):
        state["topic"] = "exam"

    # -------------------------
    # SMART CONVERSATION FLOW
    # -------------------------

    # First time stress
    if state["emotion"] == "stress":

        # Step 1: Ask cause
        if "what's stressing you" not in context:
            return "That sounds really overwhelming… 💛 You've been handling a lot. What’s stressing you the most?"

        # Step 2: If exams
        elif state["topic"] == "exam":

            if "hardest" not in context:
                return "Exams can feel really heavy… 📚 What part feels the hardest for you?"

            else:
                return "I understand… that pressure builds up 😔 You're trying your best. Have you been getting enough rest?"

        # Step 3: general follow-up
        else:
            return "I'm here with you 💙 You don’t have to go through this alone. Tell me a bit more."

    # Sad flow
    elif any(word in text for word in ["sad", "not okay"]):
        return "I'm really sorry you're feeling this way 💔 I'm here with you. Do you want to share what happened?"

    # Greeting
    elif any(word in text for word in ["hi", "hello", "hey"]):
        return "Hey… I’m here for you 💙 How are you feeling today?"

    # Default (SMART, not boring)
    else:
        return "I’m listening 💙 Tell me more… what’s been on your mind?"

# -----------------------------
# SEND BUTTON
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
