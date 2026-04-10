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

    risk = None
    risk_percentage = 0

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

        risk_percentage = proba[1] * 100

        if risk_percentage >= 70:
            risk = "High"
        elif risk_percentage >= 40:
            risk = "Medium"
        else:
            risk = "Low"

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
            risk = "Low"
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
    display: inline-block;
    width: fit-content;
    max-width: 60%;

    background-color: #2b313e;
    color: white;
    padding: 10px 15px;
    border-radius: 15px;

    margin-left: auto;   /* ⭐ PUSH RIGHT */
    margin-right: 0px;

    text-align: left;
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

if "show_chart" not in st.session_state:
    st.session_state.show_chart = True

st.markdown("---")
st.subheader("💬 Your Support Companion")

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    
     # RESET MEMORY (THIS IS MISSING)
    st.session_state.memory = {
        "topic": None,
        "emotion": None,
        "last_question": None
    }
    
    # Reset chart
    st.session_state.show_chart = True

    # FORCE REFRESH (THIS IS KEY 🔥)
    st.rerun()

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

    if last_question == "comfort" and text not in ["no", "not really"]:
        return random.choice([
            "I'm really glad you shared that 💙 What’s been on your mind?",
            "Thank you for opening up… I’m here for you 💙",
            "Take your time… tell me what you’re feeling 💙"
        ])

    if text in ["yes", "yeah", "ok", "okay"]:
        if last_question == "talk_more":
            st.session_state.memory["last_question"] = "talk_more"
            return random.choice([
                "I'm here for you 💙 Tell me what's been on your mind.",
                "Take your time… what would you like to share?",
                "You can tell me anything… I'm listening 💙",
                "I'm here with you… what’s been bothering you?",
                "Go ahead… I’m listening 💙"
            ])
        
    if text in ["no", "not really"]:
        st.session_state.memory["last_question"] = "comfort"
    
        return random.choice([
            "That’s okay… you don’t have to share anything you’re not ready to 💙",
            "No worries… we can just sit here for a moment together 💙",
            "That’s completely okay… I’m still here with you 🤍",
            "You don’t have to talk right now… I’m here whenever you feel ready 💙",
            "It’s okay to take your time… you’re not alone 💙"
        ])

    # -------------------------------
    # WORKPLACE MENTAL HEALTH LOGIC
    # ------------------------------
    
    if "work pressure" in text or "deadline" in text or "workload" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Work pressure can build up quickly… how has it been affecting you lately?"
    
    if "office politics" in text or "colleagues" in text or "team" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Workplace dynamics can be really challenging… do you feel supported in your environment?"
    
    if "burnout" in text or "exhausted" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Feeling burnt out can be really draining… have you been able to take any time for yourself?"
    
    if "no support" in text or "not supported" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Lack of support at work can feel isolating… would you like to talk about what’s been happening?"
    
    if "remote work" in text or "work from home" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Remote work can sometimes feel isolating or blur boundaries… how has your experience been?"
    
    if "work life balance" in text or "balance" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Maintaining work-life balance can be tough… do you feel like work is taking over your personal time?"
    
    if "job" in text and "stress" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Job-related stress can impact both mental and emotional well-being… what part of your work feels most stressful?"
    
    if "career" in text and "confused" in text:
        st.session_state.memory["last_question"] = "talk_more"
        return "Career uncertainty can feel overwhelming… what concerns you the most right now?"
    # DETECT TOPIC

    if "future" in text:
        st.session_state.memory["topic"] = "future"
        st.session_state.memory["last_question"] = "talk_more"
        return "Thinking about the future can feel heavy sometimes… what part worries you the most?"
    
    elif "study" in text or "exam" in text or "college" in text:
        st.session_state.memory["topic"] = "studies"
        st.session_state.memory["last_question"] = "talk_more"
        return "Studies can feel really stressful… what part of it feels most overwhelming to you?"
    
    elif "family" in text or "parents" in text:
        st.session_state.memory["topic"] = "family"
        st.session_state.memory["last_question"] = "talk_more"
        return "Family situations can be really emotional… do you want to tell me what’s going on?"

    elif "friend" in text:
         st.session_state.memory["topic"] = "friendship"
         st.session_state.memory["last_question"] = "talk_more"
         return "Friendship issues can really hurt...do you feellike you're being ignored or misunderstood?"
    
    elif "alone" in text or "lonely" in text:
        st.session_state.memory["topic"] = "loneliness"
        st.session_state.memory["last_question"] = "talk_more"
        return "Feeling alone can be really hard… do you feel like you don’t have someone to talk to?"
    
    elif "relationship" in text or "love" in text:
        st.session_state.memory["topic"] = "relationship"
        st.session_state.memory["last_question"] = "talk_more"
        return "Relationships can be confusing sometimes… what’s been on your mind?"

    # REMEMBER PREVIOUS TOPIC

    previous_topic = st.session_state.memory.get("topic")
    
    if previous_topic and previous_topic in text:
        return f"Earlier you mentioned {previous_topic}… is that still bothering you?"

    # CONTEXT-AWARE FOLLOW-UP

    topic = st.session_state.memory.get("topic")
    
    if topic == "future":
        return random.choice([
            "It’s okay to feel uncertain about the future… is it about career or something else?",
            "The future can feel scary… what part worries you the most right now?"
        ])
    
    elif topic == "studies":
        return random.choice([
            "Studies can really build pressure… is it exams or understanding subjects?",
            "That sounds stressful… are you feeling overwhelmed with workload?"
        ])
    
    elif topic == "family":
        return random.choice([
            "Family situations can be tough… is it something someone said or ongoing stress?",
            "I understand… family issues can feel heavy. Do you want to share what happened?"
        ])
    
    elif topic == "loneliness":
        return random.choice([
            "Feeling lonely can be really painful… do you feel left out or disconnected?",
            "I’m here with you… when do you feel this loneliness the most?"
        ])
    
    elif topic == "relationship":
        return random.choice([
            "Relationships can be emotionally draining… what’s been bothering you?",
            "Do you feel confused or hurt in this situation?"
        ])
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
            "Can you share a bit more about that?",
            "What part of this feels the hardest?",
            "I'm here with you... tell me what you're thinking 💙"
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
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; width: 100%;">
            <div class="user-bubble">🧍 {msg}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; width: 100%;">
            <div class="bot-bubble">🤖 {msg}</div>
        </div>
        """, unsafe_allow_html=True)

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

def get_recommendation(risk, risk_percentage):
    if risk == "High":
        return {
            "message": "⚠️ Immediate attention needed. Reduce workload and seek support.",
            "impact": f"Risk level is high ({risk_percentage:.2f}%)",
            "extra": "Consider professional help and company support programs"
        }
    elif risk == "Medium":
        return {
            "message": "⚡ Try improving work-life balance and reduce stress",
            "impact": f"Moderate risk detected ({risk_percentage:.2f}%)",
            "extra": "Engage in wellness activities and talk to someone"
        }
    else:
        return {
            "message": "✅ Maintain your current healthy routine",
            "impact": f"Low risk ({risk_percentage:.2f}%)",
            "extra": "Continue good habits and stay connected"
        }

# ✅ Use REAL prediction
rec = get_recommendation(risk, risk_percentage)

st.write("Risk Level:", risk)
st.write("Suggestion:", rec["message"])
st.write("Impact:", rec["impact"])
st.write("Extra Support:", rec["extra"])
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
