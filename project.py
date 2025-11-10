import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import requests

# Model: You may replace with your preferred Indic LLM on Hugging Face Hub
MODEL_NAME = "ai4bharat/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
fact_check_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Supported Indian languages (add/remove language codes as needed)
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Bhojpuri": "bho"
}

def fetch_evidence(claim: str, lang_code: str) -> str:
    """Retrieve top Wikipedia snippets as evidence for the claim."""
    url = f"https://{lang_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": claim,
        "format": "json",
    }
    response = requests.get(url, params=params)
    results = response.json().get("query", {}).get("search", [])
    evidence = " ".join([r["snippet"] for r in results[:3]])
    return evidence

def fact_check_claim(claim: str, lang_code: str) -> str:
    """Generate fact-checking verdict and explanation with the LLM."""
    evidence = fetch_evidence(claim, lang_code)
    prompt = f"Claim: {claim}\nEvidence: {evidence}\nTask: Fact-check this claim in language code {lang_code}. Respond with True/False/Unverifiable and explanation."
    result = fact_check_pipeline(prompt, max_length=256)[0]['generated_text']
    return result

# Streamlit UI
st.title("Multilingual Fact-Checking for Indian Languages")

language = st.selectbox("Select Language", options=list(LANGUAGES.keys()))
claim = st.text_area("Input Claim", height=100)

if st.button("Fact Check"):
    if claim.strip():
        lang_code = LANGUAGES[language]
        with st.spinner("Fact-checking in progress..."):
            verdict = fact_check_claim(claim, lang_code)
        st.markdown("### Verdict and Explanation")
        st.write(verdict)
    else:
        st.warning("Please enter a claim to fact-check.")
