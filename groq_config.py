import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)

# Task 1: Extract Info
system_prompt_task1 = """
You are an experienced NLP and healthcare expert working on medical transcription summarization.
You will be given a transcript between a doctor and a patient. Your task is to extract a structured JSON object with the following 6 fields:

1. "Patient_Name" – Extract the patient's name from the conversation
2. "Symptoms" – List clearly mentioned or implied symptoms.
3. "Diagnosis" – Use medical reasoning to infer a likely diagnosis.
4. "Treatment" – Include all mentioned treatments, and intelligently infer standard treatments (e.g., painkillers for pain).
5. "Current_Status" – Describe current health based on latest patient comment.
6. "Prognosis" – Predict the likely recovery or outcome based on symptoms and treatment, even if not explicitly stated.

💡 Use proper medical terminology and return a valid JSON object ONLY — no notes or explanation.
"""

def extract_info(conversation):
    user_prompt = f"""Extract the following fields from this transcript:\n{conversation}"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt_task1},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


# Task 2: Sentiment & Intent Detection
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


# Task 3: SOAP Note
def generate_soap_note(transcript):
    prompt = f"""
You are an intelligent medical assistant. Your task is to analyze a transcript of a doctor-patient conversation and produce a SOAP note in structured JSON format.

Follow these strict instructions:

1. Extract information explicitly mentioned in the conversation.
2. If something is **not mentioned**, write `"Not mentioned"` clearly—do not assume or invent.
3. Format your response as a **JSON object with the following keys**: Subjective, Objective, Assessment, and Plan.
4. Include these specific subfields:

**Subjective:**
- Chief_Complaint: What the patient says is bothering them.
- History_of_Present_Illness: Timeline, cause, or relevant history in patient's own words.

**Objective:**
- Physical_Exam_Findings: Note exam observations (e.g., tenderness, swelling). If not mentioned, say "Not mentioned".
- Clinical_Facts: Include any facts like treatments, vitals, lab results, or test findings.

**Assessment:**
- Diagnosis: Use the patient's condition in medical terms if possible.
- Severity: Use terms like "Mild", "Moderate", "Severe", or "Not mentioned".

**Plan:**
- Treatment: Ongoing or recommended actions like physiotherapy or rest.
- Medication: Mention drugs or say "Not mentioned".
- Follow_up_Plan: Time or condition for next check-up or escalation.

Respond only with a valid JSON object. Do not explain or include anything outside the JSON.

Transcript:
\"\"\"{transcript}\"\"\"
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content
