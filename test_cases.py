import pandas as pd
from app import model, encoders, meta, FEATURES, OPTIONS

AGREE_SCORED = {1: True, 2: False, 3: False, 4: False,
                5: True, 6: True, 7: False, 8: False, 9: True, 10: False}


def aq_answers(scored_map):
    """scored_map: {question_id: want_point} -> choice per question"""
    choices = {}
    for qid, want in scored_map.items():
        if AGREE_SCORED[qid] == want:
            choices[qid] = "da"
        else:
            choices[qid] = "dd"
    return choices


def total_score(choices):
    return sum(1 for qid, c in choices.items()
               if (c == "da") == AGREE_SCORED[qid])


def encode_feature(column, value):
    encoder = encoders[column]
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return int(encoder.transform(["others"])[0])


def predict(case_name, choices, age, gender, ethnicity, jaundice,
            austim, country, used_app_before, relation):
    score = total_score(choices)
    record = {f"A{i}_Score": int((choices[i] == "da") == AGREE_SCORED[i]) for i in range(1, 11)}
    record.update({
        "age": age, "gender": gender, "ethnicity": ethnicity,
        "jaundice": jaundice, "austim": austim, "contry_of_res": country,
        "used_app_before": used_app_before, "result": float(score),
        "relation": relation,
    })
    vector = [encode_feature(f, record[f]) if f in encoders else record[f]
              for f in FEATURES]
    frame = pd.DataFrame([vector], columns=FEATURES)
    pred = int(model.predict(frame)[0])
    proba = model.predict_proba(frame)[0]
    classes = [int(c) for c in model.classes_]
    conf = float(proba[classes.index(pred)]) * 100
    label = "ASD DETECTED" if pred == 1 else "NO INDICATORS"
    print(f"{case_name:<28} AQ={score:>2}/10  ->  {label:<14} ({conf:.1f}% confident)")


print("=" * 70)
print("VERIFIED TEST CASES (actual model output)")
print("=" * 70)

trait = {q: True for q in range(1, 11)}
no_trait = {q: False for q in range(1, 11)}
border6 = dict(no_trait); border6.update({1: True, 2: True, 5: True, 6: True, 9: True, 10: True})
border5 = dict(no_trait); border5.update({1: True, 5: True, 6: True, 9: True, 10: True})
mid8 = dict(no_trait); mid8.update({1: True, 2: True, 5: True, 6: True, 8: True, 9: True, 10: True})

c1 = aq_answers(trait)
predict("TC1 max score + risk factors", c1, 21, "m",
        "White-European", "no", "yes", "United States", "no", "Self")

c2 = aq_answers(no_trait)
predict("TC2 zero score, low risk", c2, 35, "f",
        "Asian", "no", "no", "India", "no", "Self")

c3 = aq_answers(border6)
predict("TC3 borderline score 6", c3, 29, "m",
        "Middle Eastern ", "no", "no", "Jordan", "no", "Self")

c4 = aq_answers(border5)
predict("TC4 sub-threshold score 5", c4, 41, "f",
        "Latino", "no", "no", "Mexico", "no", "Self")

c5 = aq_answers(mid8)
predict("TC5 high score 7, child via parent", c5, 4, "m",
        "South Asian", "yes", "yes", "India", "yes", "others")

c6 = aq_answers(trait)
predict("TC6 max score, female", c6, 27, "f",
        "others", "no", "no", "United Kingdom", "yes", "Self")

c7 = aq_answers(no_trait)
predict("TC7 zero score, family ASD", c7, 19, "m",
        "Black", "no", "yes", "United States", "no", "Self")

c8 = aq_answers(border6)
predict("TC8 borderline, elderly", c8, 64, "f",
        "White-European", "no", "no", "New Zealand", "no", "others")

print()
print("Ethnicity options available:", OPTIONS["ethnicity"])
