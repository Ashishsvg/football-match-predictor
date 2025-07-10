import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="centered")

df = pd.read_csv("C:/Users/Ashish/Desktop/DSBDA/DSBDA Mini project/results.csv")

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "Win"
    elif row["home_score"] == row["away_score"]:
        return "Draw"
    else:
        return "Loss"

df["match_result"] = df.apply(get_result, axis=1)

if "neutral" in df.columns:
    df = df.drop(columns=["neutral"])

categorical_cols = ["home_team", "away_team", "tournament", "city", "country"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

result_encoder = LabelEncoder()
df["match_result"] = result_encoder.fit_transform(df["match_result"])

X = df[categorical_cols]
y = df["match_result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = LogisticRegression(max_iter=4000, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.title("⚽ Football Match Outcome Predictor")
st.markdown("---")
st.markdown("Use historical match data and machine learning to predict the winner of a football match!")

st.subheader("🔢 Select Teams")

home_teams = sorted(encoders["home_team"].classes_)
away_teams = sorted(encoders["away_team"].classes_)

home_team = st.selectbox("🏠 Home Team", home_teams)
away_team = st.selectbox("🚗 Away Team", away_teams)

if st.button("🔮 Predict Result"):
    try:
        home_encoded = encoders["home_team"].transform([home_team])[0]
        away_encoded = encoders["away_team"].transform([away_team])[0]
    except ValueError:
        st.error("Invalid team name.")
        st.stop()

    most_common = df.mode().iloc[0]
    input_data = {
        "home_team": home_encoded,
        "away_team": away_encoded,
        "tournament": most_common["tournament"],
        "city": most_common["city"],
        "country": most_common["country"]
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    predicted_result = result_encoder.inverse_transform(prediction)[0]

    st.markdown("## ✅ Predicted Match Outcome")
    if predicted_result == "Win":
        st.success(f"🏆 **{home_team}** is predicted to **Win**!")
    elif predicted_result == "Loss":
        st.success(f"🏆 **{away_team}** is predicted to **Win**!")
    else:
        st.info("⚖️ It's a predicted **Draw**")

    st.markdown(f"### 📈 Model Accuracy: `{acc*100:.2f}%`")

    st.markdown("---")
    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=result_encoder.classes_,
                yticklabels=result_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
