from groq_config import client

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
\"\"\"
{transcript}
\"\"\"
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content
