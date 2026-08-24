import json
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "best_model.pkl", "rb") as handle:
    model = pickle.load(handle)
with open(MODEL_DIR / "encoders.pkl", "rb") as handle:
    encoders = pickle.load(handle)
with open(MODEL_DIR / "meta.json", "r", encoding="utf-8") as handle:
    meta = json.load(handle)
with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as handle:
    metrics = json.load(handle)

FEATURES = meta["features"]
OPTIONS = meta["options"]

QUESTION_ITEMS = [
    {"id": 1, "text": "I often notice small sounds when others do not.", "scores_on_agree": True},
    {"id": 2, "text": "I usually concentrate more on the overall picture, rather than the small details.", "scores_on_agree": False},
    {"id": 3, "text": "I find it easy to do more than one thing at once.", "scores_on_agree": False},
    {"id": 4, "text": "If there is an interruption, I can switch back to what I was doing very quickly.", "scores_on_agree": False},
    {"id": 5, "text": "I frequently get so strongly absorbed in one thing that I lose sight of other things.", "scores_on_agree": True},
    {"id": 6, "text": "I often notice details that others do not.", "scores_on_agree": True},
    {"id": 7, "text": "I find it easy to go back and forth between different activities.", "scores_on_agree": False},
    {"id": 8, "text": "I know how to tell if someone listening to me is getting bored.", "scores_on_agree": False},
    {"id": 9, "text": "When I'm reading a story, I find it difficult to work out the characters' intentions.", "scores_on_agree": True},
    {"id": 10, "text": "I find it easy to work out people's intentions.", "scores_on_agree": False},
]

AGREE_VALUES = {"da", "sa"}

app = Flask(__name__)


def encode_feature(column, value):
    encoder = encoders[column]
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    fallback = "others" if "others" in encoder.classes_ else encoder.classes_[0]
    return int(encoder.transform([fallback])[0])


def collect_answers(form):
    selections = []
    total_score = 0
    for item in QUESTION_ITEMS:
        choice = form.get(f"q{item['id']}")
        if choice not in {"da", "sa", "sd", "dd"}:
            return None, None
        agrees = choice in AGREE_VALUES
        scored = int(agrees == item["scores_on_agree"])
        selections.append({"item": item, "choice": choice, "scored": scored})
        total_score += scored
    return selections, total_score


@app.route("/")
def home():
    return render_template("index.html", meta=meta, metrics=metrics)


@app.route("/assess", methods=["GET", "POST"])
def assess():
    if request.method == "POST":
        background = extract_background_raw(request.form)
        saved_answers = extract_saved_answers(request.form)
    else:
        background = None
        saved_answers = None
    return render_template(
        "background.html",
        meta=meta,
        background=background,
        saved_answers=saved_answers,
    )


def extract_saved_answers(form):
    return {f"q{i}": form.get(f"q{i}", "") for i in range(1, 11)}


def extract_background_raw(form):
    return {
        "age": str(form.get("age", "")).strip(),
        "gender": form.get("gender", ""),
        "ethnicity": form.get("ethnicity", ""),
        "jaundice": form.get("jaundice", ""),
        "austim": form.get("austim", ""),
        "contry_of_res": form.get("contry_of_res", ""),
        "used_app_before": form.get("used_app_before", ""),
        "relation": form.get("relation", ""),
    }


def parse_background(form):
    raw_age = str(form.get("age", "")).strip()
    try:
        age = int(raw_age)
    except (TypeError, ValueError):
        age = -1
    if not 1 <= age <= 120:
        return None, "Please enter a valid whole-number age between 1 and 120."

    gender = form.get("gender", "")
    ethnicity = form.get("ethnicity", "")
    jaundice = form.get("jaundice", "")
    austim = form.get("austim", "")
    country = form.get("contry_of_res", "")
    used_app_before = form.get("used_app_before", "")
    relation = form.get("relation", "")

    required = {
        "gender": gender in OPTIONS["gender"],
        "ethnicity": ethnicity in OPTIONS["ethnicity"],
        "jaundice": jaundice in OPTIONS["jaundice"],
        "austim": austim in OPTIONS["austim"],
        "contry_of_res": country in OPTIONS["contry_of_res"],
        "used_app_before": used_app_before in OPTIONS["used_app_before"],
        "relation": relation in OPTIONS["relation"],
    }
    if not all(required.values()):
        return None, "Please complete every field before continuing."

    return {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jaundice": jaundice,
        "austim": austim,
        "contry_of_res": country,
        "used_app_before": used_app_before,
        "relation": relation,
    }, None


@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():
    if request.method == "GET":
        return redirect(url_for("assess"))

    background, error = parse_background(request.form)
    if error:
        return render_template(
            "background.html",
            meta=meta,
            background=extract_background_raw(request.form),
            saved_answers=extract_saved_answers(request.form),
            error=error,
        )

    return render_template(
        "questionnaire.html",
        meta=meta,
        questions=QUESTION_ITEMS,
        background=background,
        saved_answers=extract_saved_answers(request.form),
    )


@app.route("/result", methods=["POST"])
def result():
    background, bg_error = parse_background(request.form)
    if bg_error:
        return render_template(
            "background.html",
            meta=meta,
            background=extract_background_raw(request.form),
            saved_answers=extract_saved_answers(request.form),
            error=bg_error,
        )

    selections, total_score = collect_answers(request.form)
    if selections is None:
        return render_template(
            "questionnaire.html",
            meta=meta,
            questions=QUESTION_ITEMS,
            background=background,
            saved_answers=extract_saved_answers(request.form),
            error="Please answer all ten screening questions before submitting.",
        )

    record = {f"A{i}_Score": selections[i - 1]["scored"] for i in range(1, 11)}
    record.update({**background, "result": float(total_score)})

    vector = []
    for feature in FEATURES:
        if feature in encoders:
            vector.append(encode_feature(feature, record[feature]))
        else:
            vector.append(record[feature])

    input_frame = pd.DataFrame([vector], columns=FEATURES)
    prediction = int(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]
    classes = [int(class_label) for class_label in model.classes_]
    positive_probability = float(probabilities[classes.index(1)])
    confidence = float(probabilities[classes.index(prediction)])

    summary = [
        ("Age", background["age"]),
        ("Gender", "Male" if background["gender"] == "m" else "Female"),
        ("Ethnicity", "Others / Unspecified" if background["ethnicity"] == "others" else background["ethnicity"].strip()),
        ("Country of residence", background["contry_of_res"]),
        ("Born with jaundice", "Yes" if background["jaundice"] == "yes" else "No"),
        ("Immediate family with ASD", "Yes" if background["austim"] == "yes" else "No"),
        ("Used a screening app before", "Yes" if background["used_app_before"] == "yes" else "No"),
        ("Completed by", "Myself" if background["relation"] == "Self" else "Someone else"),
    ]

    return render_template(
        "result.html",
        meta=meta,
        prediction=prediction,
        confidence=confidence,
        positive_probability=positive_probability,
        total_score=total_score,
        selections=selections,
        summary=summary,
    )


@app.route("/about")
def about():
    return render_template("about.html", meta=meta, metrics=metrics)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("RENDER") is None)
