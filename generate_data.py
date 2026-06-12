import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Symptom scores from Prolog KB (exact match)
symptom_scores = {
    'low_weight': 3, 'stunted_growth': 4, 'swollen_belly': 5, 'frequent_illness': 3,
    'pale_skin': 2, 'fatigue': 2, 'brittle_hair': 2, 'delayed_development': 4,
    'skin_infections': 2, 'weak_immunity': 3, 'hair_discoloration': 2,
    'loss_of_appetite': 1, 'irritability': 1, 'diarrhea': 2, 'slow_healing_wounds': 2
}
symptom_names = list(symptom_scores.keys())

# 5 types (aligned to Prolog intent: severe, moderate, chronic, mild, normal; 40 cases each for 200 total)
malnutrition_types = [
    ('severe_acute_malnutrition', 90),
    ('moderate_acute_malnutrition', 60),
    ('chronic_malnutrition', 120),  # Base days higher for long-term
    ('mild_malnutrition', 30),
    ('normal', 15)
]
n_cases_per_type = 40  # Even: 40 * 5 = 200

# Symptom biases (adjusted to include chronic-specific)
type_symptom_bias = {
    'severe_acute_malnutrition': ['swollen_belly', 'frequent_illness', 'low_weight', 'fatigue'],
    'moderate_acute_malnutrition': ['stunted_growth', 'pale_skin', 'low_weight'],
    'chronic_malnutrition': ['delayed_development', 'brittle_hair', 'skin_infections', 'weak_immunity'],  # Long-term indicators
    'mild_malnutrition': ['loss_of_appetite', 'irritability', 'hair_discoloration', 'diarrhea', 'slow_healing_wounds'],
    'normal': []
}

# Simulation of Prolog (with age split for moderate/chronic distinction; actual Prolog will map chronic to moderate)
def prolog_malnutrition_type(sc, ss, age):
    if sc >= 4 and ss >= 12:
        return 'severe_acute_malnutrition'
    elif sc >= 2 and ss >= 6 and ss < 12:
        if age <= 7:  # Distinguish acute (younger) vs. chronic (older) for simulation/ground truth
            return 'moderate_acute_malnutrition'
        else:
            return 'chronic_malnutrition'  # Intended; actual Prolog without age will call this 'moderate'
    elif sc >= 1 and ss >= 1:
        return 'mild_malnutrition'
    else:
        return 'none'

def prolog_weight_category(age, weight):
    if (age <= 2 and weight < 8) or (2 < age <= 5 and weight < 12) or (5 < age <= 10 and weight < 18) or (10 < age <= 12 and weight < 24):
        return 'severely_underweight'
    elif (age <= 2 and 8 <= weight < 10) or (2 < age <= 5 and 12 <= weight < 15) or (5 < age <= 10 and 18 <= weight < 22) or (10 < age <= 12 and 24 <= weight < 30):
        return 'moderately_underweight'
    else:
        return 'normal'

def prolog_severity_level(ss):
    if ss <= 6:
        return 'mild'
    elif 6 < ss <= 12:
        return 'moderate'
    else:
        return 'severe'

# Generate data (non-overlapping; age split for moderate/chronic)
data = []
for type_idx in range(len(malnutrition_types)):
    maln_type, base_days = malnutrition_types[type_idx]
    for _ in range(n_cases_per_type):
        # Age: Split for moderate/chronic (others uniform)
        if maln_type == 'moderate_acute_malnutrition':
            age = np.random.uniform(1, 7.0)  # <=7 for acute moderate
        elif maln_type == 'chronic_malnutrition':
            age = np.random.uniform(7.1, 12)  # >7 for chronic (older/long-term)
        else:
            age = np.random.uniform(1, 12)  # Uniform for others
        age = round(np.clip(age, 1, 12), 1)
        
        # Expected weight
        expected_weight = age * 4.5
        
        # Weight ratio: Tuned per type (chronic slightly higher than moderate for persistence)
        if maln_type == 'severe_acute_malnutrition':
            weight_ratio = np.random.uniform(0.35, 0.55)
        elif maln_type == 'moderate_acute_malnutrition':
            weight_ratio = np.random.uniform(0.60, 0.75)
        elif maln_type == 'chronic_malnutrition':
            weight_ratio = np.random.uniform(0.70, 0.85)  # Slightly better than moderate (chronic but stable)
        elif maln_type == 'mild_malnutrition':
            weight_ratio = np.random.uniform(0.80, 0.95)
        else:  # normal
            weight_ratio = np.random.uniform(0.95, 1.05)
        weight_ratio += np.random.normal(0, 0.005)
        weight_ratio = np.clip(weight_ratio, 0.3, 1.1)
        weight_ratio = round(weight_ratio, 2)
        
        weight = expected_weight * weight_ratio
        weight = round(np.clip(weight, 3, 40), 1)
        
        # Symptom count: Aligned to thresholds (non-overlapping where possible)
        if maln_type == 'severe_acute_malnutrition':
            symptom_count = np.random.randint(4, 10)  # >=4
        elif maln_type in ['moderate_acute_malnutrition', 'chronic_malnutrition']:
            symptom_count = np.random.randint(3, 5)  # >=3 (covers both; SC=2 avoided to lean chronic-like)
        elif maln_type == 'mild_malnutrition':
            symptom_count = np.random.randint(1, 2)  # >=1 <3
        else:  # normal
            symptom_count = 0
        
        # Symptoms: Bias to type, exact count
        selected_symptoms = []
        if maln_type != 'normal' and type_symptom_bias[maln_type]:
            key_syms = type_symptom_bias[maln_type][:min(2, len(type_symptom_bias[maln_type]))]
            selected_symptoms.extend(key_syms)
            remaining = max(0, symptom_count - len(key_syms))
            if remaining > 0:
                other = [s for s in symptom_names if s not in key_syms]
                selected_symptoms += random.sample(other, min(remaining, len(other)))
        
        while len(selected_symptoms) < symptom_count:
            avail = [s for s in symptom_names if s not in selected_symptoms]
            if avail:
                selected_symptoms.append(random.choice(avail))
        
        selected_symptoms = selected_symptoms[:symptom_count]
        
        # Severity score: Aligned bins (chronic/moderate overlap in Prolog, but split by age here)
        base_ss = sum(symptom_scores[s] for s in selected_symptoms)
        if maln_type == 'severe_acute_malnutrition':
            severity_score = round(np.clip(base_ss + np.random.uniform(2, 5), 12, 25), 1)  # >=12
        elif maln_type in ['moderate_acute_malnutrition', 'chronic_malnutrition']:
            severity_score = round(np.clip(base_ss + np.random.uniform(2, 4), 8, 11.9), 1)  # >=8 <12 (chronic-like; Prolog will call moderate)
        elif maln_type == 'mild_malnutrition':
            severity_score = round(np.clip(base_ss + np.random.uniform(0, 1), 1, 5.9), 1)  # 1-5.9
        else:  # normal
            severity_score = 0.0
        
        # Verify simulation (uses age split; actual Prolog ignores age, maps chronic to moderate)
        verified_prolog_type = prolog_malnutrition_type(symptom_count, severity_score, age)
        verified_weight_cat = prolog_weight_category(age, weight)
        verified_sev_level = prolog_severity_level(severity_score)
        
        # Outcome
        noise = np.random.normal(0, base_days * 0.02)
        weight_penalty = max(0, (1 - weight_ratio) * 6)
        outcome_days = base_days + noise + weight_penalty
        outcome_days = round(np.clip(outcome_days, 10, 150), 1)
        
        data.append({
            'age': age,
            'weight': weight,
            'weight_ratio': weight_ratio,
            'symptom_count': symptom_count,
            'severity_score': severity_score,
            'outcome_days': outcome_days,
            'malnutrition_type': maln_type,
            'prolog_group': verified_prolog_type,  # Simulated with age; actual Prolog: chronic → moderate
            'weight_category': verified_weight_cat,
            'severity_level': verified_sev_level,
            'selected_symptoms': '|'.join(selected_symptoms)
        })

# DataFrame and CSV
df = pd.DataFrame(data)
df.to_csv('malnutrition_cases_improved.csv', index=False)

# Summary
print("Generated 'malnutrition_cases_improved.csv' (5 types including chronic, 40 each; note: Prolog maps chronic to moderate).")
print(f"Shape: {df.shape}")
print("\nDistribution by Ground Truth (malnutrition_type):")
print(df['malnutrition_type'].value_counts().sort_index())

print("\nDistribution by Simulated Prolog Group (with age split; actual Prolog: chronic → moderate_acute_malnutrition):")
print(df['prolog_group'].value_counts().sort_index())

print("\nAlignment Crosstab (simulated; 100% diagonal):")
print(pd.crosstab(df['malnutrition_type'], df['prolog_group']))

print("\nAverage Features by Type (non-overlapping except chronic/moderate SS overlap):")
print(df.groupby('malnutrition_type')[['weight_ratio', 'symptom_count', 'severity_score', 'age']].agg(['mean', 'min', 'max']).round(2))

print("\nProlog Weight Categories by Type:")
print(df.groupby('malnutrition_type')['weight_category'].value_counts())

print("\nProlog Severity Levels by Type:")
print(df.groupby('malnutrition_type')['severity_level'].value_counts())

print("\nSample (one per type):")
for typ in malnutrition_types:
    sample = df[df['malnutrition_type'] == typ[0]].iloc[0]
    actual_prolog_note = " (actual Prolog: moderate_acute_malnutrition)" if typ[0] == 'chronic_malnutrition' else ""
    print(f"{typ[0]}: Age={sample['age']}, Ratio={sample['weight_ratio']}, SC={sample['symptom_count']}, SS={sample['severity_score']}, Simulated Prolog={sample['prolog_group']}{actual_prolog_note}, Weight Cat={sample['weight_category']}, Sev Level={sample['severity_level']}, Days={sample['outcome_days']}")
