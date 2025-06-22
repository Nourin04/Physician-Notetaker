from groq_config import client

def analyze_patient_input(text):
    prompt = f"""
You are a medical AI assistant. Given a patient’s message, perform the following tasks:

1. Classify the overall **Sentiment** of the message into one of the following:
   - Anxious: Shows concern, fear, or worry about their health.
   - Neutral: Simply provides information or makes a request without emotion.
   - Reassured: Shows relief, satisfaction, or trust after receiving care or explanation.

2. Identify the **Intent** behind the message in 3–5 words. Use one of the following or your best judgment:
   - Reporting symptoms
   - Seeking reassurance
   - Expressing concern
   - Scheduling appointment
   - Requesting information
   - Providing feedback
   - Expressing gratitude
   - Asking for treatment
   - Describing progress
   - Providing lifestyle information

Output only valid JSON as:
{{
  "Sentiment": "...",
  "Intent": "..."
}}

Patient Dialogue: "{text}"
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content
