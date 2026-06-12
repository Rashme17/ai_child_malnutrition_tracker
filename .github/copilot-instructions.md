# Copilot Instructions for Q-Learning Malnutrition Diagnosis System

## Project Overview
This is a hybrid intelligent system combining Q-Learning, Prolog-based expert system, and machine learning for malnutrition diagnosis and treatment recommendations. The system uses:
- Flask web interface (`app.py`)
- Q-Learning agent for treatment recommendations (`q_learning_agent.py`)
- Prolog knowledge base for diagnosis rules (`knowledge.pl`)
- Machine learning models for prediction and clustering

## Key Architecture Components

### 1. Core Components
- **Expert System**: Prolog-based diagnosis engine in `knowledge.pl`
  - Uses dynamic facts for symptoms, age, weight
  - Symptom catalog with severity scoring
  - Malnutrition outcome classifications

- **Q-Learning Agent** (`q_learning_agent.py`):
  - Learns optimal treatment paths
  - Uses state-action-reward mapping
  - Supports feedback-based learning (approve/disapprove)

- **Web Interface** (`app.py`):
  - Flask routes for diagnosis and treatment
  - Visualization of diagnosis graphs
  - Integration of all components

### 2. Data Flow
1. User inputs → Flask → Prolog engine
2. Diagnosis results → Q-Learning agent
3. Treatment recommendations → User feedback → Agent learning

## Development Workflows

### Model Training
- Run `train_model.py` before using ML predictions
- Model artifacts: `regression_model.pkl`, `malnutrition_model.pkl`

### Expert System Updates
- Add new symptoms to `knowledge.pl` symptom_catalog/1
- Define symptom_score/2 for severity calculation
- Update malnutrition_outcome/2 for classification thresholds

### Q-Learning Agent Modifications
- Adjust learning parameters in `q_learning_agent.py`:
  ```python
  alpha=0.5  # Learning rate
  gamma=0.9  # Discount factor
  epsilon=0.1  # Exploration rate
  ```

## Project Conventions

### Data Files
- CSV format for case data: `malnutrition_cases.csv`
- Derived datasets:
  - `malnutrition_cases_with_clusters.csv`
  - `malnutrition_cases_with_risk_clusters.csv`

### Integration Points
1. Prolog-Python Interface:
   - Uses `pyswip` for Prolog queries
   - Fact assertion pattern: `prolog.assertz(f"symptom({symptom})")`

2. Agent-Flask Communication:
   - Global `agent_trained` flag for initialization
   - Feedback system via approve/disapprove actions

### Error Handling
- Prolog KB loading validation in `consult_kb()`
- Model loading with graceful degradation
- Dynamic fact cleanup between sessions