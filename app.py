
from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Demo training data banayein
data = pd.DataFrame({
    "previous_marks": [35, 45, 50, 60, 65, 70, 75, 80, 85, 90,
                       40, 55, 68, 72, 88, 30, 62, 78, 82, 95],
    "attendance": [50, 60, 65, 70, 75, 80, 85, 90, 95, 98,
                   55, 68, 72, 78, 92, 40, 75, 88, 80, 99],
    "study_hours": [1, 2, 2, 3, 3, 4, 4, 5, 6, 7,
                    1, 2, 3, 4, 6, 1, 3, 5, 5, 8],
    "marks": [38, 46, 52, 61, 66, 72, 76, 82, 87, 93,
              42, 54, 67, 74, 89, 32, 63, 79, 84, 96]
})

X = data[["previous_marks", "attendance", "study_hours"]]
y = data["marks"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Performance Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #eef2ff;
            margin: 0;
            padding: 30px 15px;
            color: #172554;
        }
        .card {
            max-width: 520px;
            margin: auto;
            background: white;
            padding: 28px;
            border-radius: 16px;
            box-shadow: 0 8px 30px #0001;
        }
        h1 { color: #4338ca; }
        label { display: block; margin-top: 16px; }
        input {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            margin-top: 7px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }
        button {
            width: 100%;
            margin-top: 22px;
            padding: 13px;
            border: 0;
            border-radius: 8px;
            background: #4f46e5;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        .result {
            margin-top: 22px;
            padding: 18px;
            border-radius: 10px;
            background: #eef2ff;
        }
        .note { color: #64748b; font-size: 13px; }
    </style>
</head>
<body>
<div class="card">
    <h1>Student Performance Predictor</h1>
    <p>Enter student details to estimate future marks.</p>

    <form method="POST">
        <label>Previous Marks (0–100)</label>
        <input type="number" name="previous_marks"
               min="0" max="100" step="any" required>

        <label>Attendance (%)</label>
        <input type="number" name="attendance"
               min="0" max="100" step="any" required>

        <label>Daily Study Hours (0–24)</label>
        <input type="number" name="study_hours"
               min="0" max="24" step="any" required>

        <button type="submit">Predict Performance</button>
    </form>

    {% if prediction is not none %}
    <div class="result">
        <h2>Estimated Marks: {{ prediction }}/100</h2>
        <h3>Performance: {{ category }}</h3>
        <p>{{ advice }}</p>
    </div>
    {% endif %}

    <p class="note">
        Educational demo only. The model uses sample data, not real
        student records. Predictions are estimates, not guarantees.
    </p>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    category = ""
    advice = ""

    if request.method == "POST":
        previous = float(request.form["previous_marks"])
        attendance = float(request.form["attendance"])
        hours = float(request.form["study_hours"])

        features = pd.DataFrame([{
            "previous_marks": previous,
            "attendance": attendance,
            "study_hours": hours
        }])

        prediction = round(float(model.predict(features)[0]), 1)
        prediction = max(0, min(100, prediction))

        if prediction >= 80:
            category = "Excellent"
        elif prediction >= 65:
            category = "Good"
        elif prediction >= 50:
            category = "Average"
        else:
            category = "Needs Improvement"

        suggestions = []
        if attendance < 75:
            suggestions.append("Improve class attendance.")
        if hours < 2:
            suggestions.append("Try a regular daily study routine.")
        if previous < 50:
            suggestions.append("Revise previous topics and practice questions.")
        advice = " ".join(suggestions) or (
            "Keep practising and maintain your study routine."
        )

    return render_template_string(
        PAGE, prediction=prediction, category=category, advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)
