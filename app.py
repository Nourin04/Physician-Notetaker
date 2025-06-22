from flask import Flask, request, jsonify, render_template
from groq_config import extract_info, analyze_patient_input, generate_soap_note
import json  # ✅ import this
from collections import OrderedDict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")



from flask import Response
from collections import OrderedDict

@app.route("/api/task1", methods=["POST"])
def task1():
    input_text = request.json.get("input")
    raw_result = extract_info(input_text)

    try:
        data = json.loads(raw_result)

        # Force the correct order
        ordered_result = OrderedDict([
            ("Patient_Name", data.get("Patient_Name")),
            ("Symptoms", data.get("Symptoms", [])),
            ("Diagnosis", data.get("Diagnosis", "")),
            ("Treatment", data.get("Treatment", [])),
            ("Current_Status", data.get("Current_Status", "")),
            ("Prognosis", data.get("Prognosis", ""))
        ])

        # Return JSON string manually with correct order
        return Response(
            json.dumps(ordered_result, indent=2),
            mimetype='application/json'
        )

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON returned by model"})


@app.route("/api/task2", methods=["POST"])
def task2():
    input_text = request.json.get("input")
    raw_result = analyze_patient_input(input_text)

    try:
        data = json.loads(raw_result)
        ordered = OrderedDict([
            ("Sentiment", data.get("Sentiment", "Unknown")),
            ("Intent", data.get("Intent", "Unknown"))
        ])
    except json.JSONDecodeError:
        ordered = OrderedDict([
            ("error", "Invalid JSON returned by model")
        ])

    return Response(json.dumps(ordered, indent=2), mimetype='application/json')

@app.route("/api/task3", methods=["POST"])
def task3():
    input_text = request.json.get("input")
    raw_result = generate_soap_note(input_text)

    try:
        data = json.loads(raw_result)
        ordered = OrderedDict([
            ("Subjective", data.get("Subjective", {})),
            ("Objective", data.get("Objective", {})),
            ("Assessment", data.get("Assessment", {})),
            ("Plan", data.get("Plan", {}))
        ])
    except json.JSONDecodeError:
        ordered = OrderedDict([
            ("error", "Invalid JSON returned by model")
        ])

    return Response(json.dumps(ordered, indent=2), mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True)
