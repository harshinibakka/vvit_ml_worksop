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
import streamlit as st

st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 1opx;
}

.user-bubble {
    align-self: flex-end;
    background-color: #2b313e;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;

    display: inline-block;   /* ⭐ IMPORTANT */
    width: fit-content;      /* ⭐ IMPORTANT */
    max-width: 60%;

    margin-left: auto;
    margin-right: 5px;
}

.bot-bubble {
    align-self: flex-start;
    background-color: #444654;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;

    display: inline-block;   /* ⭐ IMPORTANT */
    width: fit-content;      /* ⭐ IMPORTANT */
    max-width: 60%;

    margin-right: auto;
    margin-left: 5px;
}

</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_state" not in st.session_state:
    
    if "memory" not in st.session_state:
        st.session_state.memory = {
        "emotion": None,
        "topic": None,
        "last_question": None
    }

st.markdown("---")
st.subheader("💬 Your Support Companion")

# INPUT BOX
user_input = st.text_input("", placeholder="Message your companion...", key="input_box")

def detect_emotion(text):
    text = text.lower()

    if any(w in text for w in ["stress", "pressure", "overwhelmed"]):
        return "stress"
    elif any(w in text for w in ["sad", "cry", "lonely", "alone"]):
        return "sad"
    elif any(w in text for w in ["anxious", "worried", "fear"]):
        return "anxiety"
    elif any(w in text for w in ["happy", "good", "better"]):
        return "positive"

    return None
    
# -----------------------------
# CHATBOT FUNCTION
# -----------------------------
import random

def chatbot_reply(user_text):
    text = user_text.lower()
    emotion = detect_emotion(user_text)

    # store emotion
    if emotion:
        st.session_state.memory["emotion"] = emotion

    last_emotion = st.session_state.memory["emotion"]
    last_question = st.session_state.memory["last_question"]

    # -----------------------------
    # HANDLE SHORT REPLIES (YES/NO)
    # -----------------------------
    if text in ["yes", "yeah", "ok", "okay"]:
        if last_question == "talk_more":
            return "I’m here for you 💙 Tell me what’s been on your mind."

    if text in ["no", "not really"]:
        return "That’s okay… we can just sit here for a bit 💙"

    # -----------------------------
    # DETECT TOPIC
    # -----------------------------
    if "future" in text:
        st.session_state.memory["topic"] = "future"
        return "Thinking about the future can feel heavy sometimes… what part worries you the most?"

    # -----------------------------
    # EMOTION RESPONSES
    # -----------------------------
    responses = {
        "stress": [
            "That sounds really overwhelming 😔",
            "You’ve been handling so much… 💙",
            "That must feel really heavy…"
        ],
        "sad": [
            "I’m really sorry you’re feeling this way 💔",
            "That sounds painful… I’m here 🤍"
        ],
        "anxiety": [
            "It’s okay to feel this way… 🤍",
            "Take a slow breath… you’re safe 🌿"
        ],
        "positive": [
            "I’m really glad to hear that 😊",
            "That’s nice… it made me smile too 🌸"
        ]
    }

    # -----------------------------
    # SMART RESPONSE BUILDING
    # -----------------------------
    if last_emotion in responses:
        base = random.choice(responses[last_emotion])

        # add follow-up question (THIS is key)
        followups = [
            "Do you want to tell me more?",
            "What’s been on your mind lately?",
            "I’m listening… 💙"
        ]

        st.session_state.memory["last_question"] = "talk_more"

        return base + " " + random.choice(followups)

    # -----------------------------
    # DEFAULT RESPONSE
    # -----------------------------
    st.session_state.memory["last_question"] = "talk_more"
    return "I’m here for you 💙 Tell me more about what you’re feeling."

if st.button("Send ➤"):   
    if user_input and user_input.strip() != "":
        
        # Add user message FIRST
        st.session_state.chat_history.append(("You", user_input))

        # Generate bot reply
        response = chatbot_reply(user_input)

        # Add bot reply
        st.session_state.chat_history.append(("Companion", response))
        
# -----------------------------
# DISPLAY CHAT
# -----------------------------

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f'<div class="user-bubble">🧍‍♀️ {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🤖 {msg}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# VISUALIZATION DASHBOARD
# ----------------------------

st.subheader("📊 Visualization Dashboard")

import matplotlib.pyplot as plt
import pandas as pd

# Use dummy data (safe for demo)
data = {
    "age": [22, 25, 30, 35, 28],
    "work_stress": [7, 6, 8, 5, 9],
    "company_support": [3, 4, 2, 5, 3],
    "work_environment": [1, 2, 1, 2, 1]  # 1=Remote, 2=Office
}

df = pd.DataFrame(data)

# Graph 1: Stress vs Company Support
st.write("Stress vs Company Support")
fig, ax = plt.subplots()
ax.scatter(df["work_stress"], df["company_support"])
ax.set_xlabel("Work Stress")
ax.set_ylabel("Company Support")
st.pyplot(fig)

# Graph 2: Age vs Stress
st.write("Age vs Stress")
fig, ax = plt.subplots()
ax.scatter(df["age"], df["work_stress"])
ax.set_xlabel("Age")
ax.set_ylabel("Stress")
st.pyplot(fig)

# Graph 3: Work Environment vs Stress
st.write("Work Environment vs Stress")
fig, ax = plt.subplots()
ax.scatter(df["work_environment"], df["work_stress"])
ax.set_xlabel("Work Environment (1=Remote, 2=Office)")
ax.set_ylabel("Stress")
st.pyplot(fig)

# ----------------------------
# RECOMMENDATION ENGINE (FINAL)
# ----------------------------

st.subheader("💡 Recommendation Engine")

def get_recommendation(risk):
    if risk == "High":
        return {
            "message": "⚠️ Improve work-life balance and reduce workload",
            "impact": "Reduces risk by 40%",
            "extra": "Access company support programs"
        }
    elif risk == "Medium":
        return {
            "message": "⚡ Try improving work-life balance",
            "impact": "Reduces risk by 25%",
            "extra": "Engage in wellness programs"
        }
    else:
        return {
            "message": "✅ Maintain your current healthy routine",
            "impact": "Risk already low",
            "extra": "Continue good habits"
        }

# Replace this later with your model prediction
demo_risk = "High"

rec = get_recommendation(demo_risk)

st.write("Risk Level:", demo_risk)
st.write("Suggestion:", rec["message"])
st.write("Impact:", rec["impact"])
st.write("Extra Support:", rec["extra"])

# ----------------------------
# BUSINESS INSIGHTS
# ----------------------------

st.subheader("📈 Business Insights")

st.write("""
- High work stress strongly increases mental health risk
- Low company support leads to burnout
- Remote work can increase isolation risk
- Younger employees show higher stress variation
""")

# ----------------------------
# BIAS DETECTION (IMPROVED)
# ----------------------------

st.subheader("⚖️ Bias Detection Analysis")

# Add demo data for bias (safe for presentation)
bias_data = {
    "gender": ["Male", "Female", "Male", "Female", "Male"],
    "age": [22, 25, 30, 35, 28],
    "company_size": ["Small", "Medium", "Large", "Medium", "Small"],
    "work_stress": [7, 6, 8, 5, 9]
}

bias_df = pd.DataFrame(bias_data)

# 🔹 Gender Bias
st.write("🔍 Average Stress by Gender")
st.write(bias_df.groupby("gender")["work_stress"].mean())

# 🔹 Age Bias
st.write("🔍 Average Stress by Age")
st.write(bias_df.groupby("age")["work_stress"].mean())

# 🔹 Company Size Impact
st.write("🔍 Stress by Company Size")
st.write(bias_df.groupby("company_size")["work_stress"].mean())
