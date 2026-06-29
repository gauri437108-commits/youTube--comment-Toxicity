import streamlit as st
from transformers import pipeline

# Page Settings
st.set_page_config(
    page_title="AI Toxicity Detection",
    page_icon="🛡️",
    layout="centered"
)

# Title
st.title("🛡️ AI Toxicity Detection Website")
st.write("Detect whether a comment is toxic or not using Artificial Intelligence.")

# Load AI Model
@st.cache_resource
def load_model():
    classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert"
    )
    return classifier

classifier = load_model()

# Input
comment = st.text_area(
    "Enter YouTube Comment",
    height=150,
    placeholder="Type your comment here..."
)

# Button
if st.button("Detect Toxicity"):

    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:

        result = classifier(comment)[0]

        label = result["label"]
        score = result["score"] * 100

        st.divider()

        st.subheader("Prediction Result")

        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {score:.2f}%")

        if label.lower() == "toxic":

            st.error("⚠️ Toxic Comment Detected")

            polite = comment

            replacements = {
                "stupid": "unwise",
                "idiot": "person",
                "hate": "dislike",
                "ugly": "different",
                "dumb": "less experienced",
                "fool": "friend"
            }

            for old, new in replacements.items():
                polite = polite.replace(old, new)
                polite = polite.replace(old.capitalize(), new.capitalize())

            st.subheader("Suggested Polite Comment")
            st.success(polite)

        else:

            st.success("✅ This comment is Non-Toxic.")

st.divider()

st.caption("Developed using Python, Streamlit and Hugging Face Transformers")
