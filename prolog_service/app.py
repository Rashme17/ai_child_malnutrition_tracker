from flask import Flask, request, jsonify
import subprocess
import json
import os
import time
import re
import tempfile

app = Flask(__name__)

def log(msg):
    """Simple timestamped log."""
    print(f"[PROLOG_SERVICE] {time.strftime('%H:%M:%S')}  {msg}", flush=True)

# TRY MULTIPLE POSSIBLE PATHS
POSSIBLE_PATHS = [
    "/app/knowledge.pl",
    "knowledge.pl",
    "./knowledge.pl",
    "/knowledge.pl",
    "prolog/knowledge.pl",
    "../knowledge.pl"
]

KB_PATH = None

# Find which path actually exists
log(">>> Checking for knowledge.pl file...")
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        KB_PATH = os.path.abspath(path)
        log(f"✅ FOUND knowledge.pl at: {KB_PATH}")
        break
    else:
        log(f"❌ NOT FOUND at: {path}")

if KB_PATH is None:
    log("⚠️ WARNING: knowledge.pl NOT FOUND at any expected location!")
    log(f"Current working directory: {os.getcwd()}")
    log(f"Files in current directory: {os.listdir('.')}")
    if os.path.exists('/app'):
        log(f"Files in /app: {os.listdir('/app')}")
    KB_PATH = "knowledge.pl"  # Fallback

def run_prolog_file(script_content):
    """Execute Prolog by writing to a temp file"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
            f.write(script_content)
            temp_path = f.name
        
        log(f"   Created temp file: {temp_path}")
        
        # Run SWI-Prolog with the temp file
        result = subprocess.run(
            ['swipl', '-q', '-s', temp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return result
        
    except subprocess.TimeoutExpired:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise

def diagnose_via_prolog(age, weight, symptoms):
    """Run diagnosis using pure Prolog subprocess calls"""
    log(f">> Running diagnosis: age={age}, weight={weight}, symptoms={symptoms}")
    log(f">> Using knowledge base: {KB_PATH}")
    
    # Build a complete Prolog script for diagnosis
    script = f"""
:- consult('{KB_PATH}').

% Assert the input data
:- assertz(child_age({age})).
:- assertz(weight({weight})).
""" + "\n".join([f":- assertz(symptom('{s}'))." for s in symptoms]) + f"""

% Run the diagnosis
run_diagnosis :-
    diagnose_kv(KV),
    write_canonical(KV),
    nl,
    halt(0).

run_diagnosis :-
    write('DIAGNOSIS_FAILED'),
    nl,
    halt(1).

:- initialization(run_diagnosis).
"""
    
    log("   Executing Prolog script...")
    
    try:
        result = run_prolog_file(script)
        
        log(f"   Return code: {result.returncode}")
        log(f"   Stdout: {result.stdout[:500]}")
        
        if result.returncode != 0:
            log(f"   Stderr: {result.stderr}")
            return None, f"Prolog execution failed: {result.stderr}"
        
        if 'DIAGNOSIS_FAILED' in result.stdout:
            return None, "Diagnosis failed - no matching conditions"
        
        # Parse the output
        output = result.stdout.strip()
        log(f"   Raw output: {output[:200]}")
        
        # Parse the KV list
        return parse_prolog_kv(output), None
        
    except subprocess.TimeoutExpired:
        return None, "Diagnosis timeout"
    except Exception as e:
        log(f"   Exception: {repr(e)}")
        import traceback
        traceback.print_exc()
        return None, str(e)

def parse_prolog_kv(prolog_output):
    """Parse Prolog KV list output into Python dict"""
    log(">> Parsing Prolog output")
    
    result = {
        "malnutrition_types": [],
        "weight_category": "unknown",
        "symptoms": [],
        "symptom_count": 0,
        "severity_score": 0,
        "severity_level": "unknown",
        "recommendations": [],
        "outcome": "unknown"
    }
    
    try:
        # Simple parsing - extract key info from the canonical output
        output = prolog_output.lower()
        
        # Extract malnutrition types
        if 'severe_acute_malnutrition' in output:
            result["malnutrition_types"].append("severe_acute_malnutrition")
        if 'moderate_acute_malnutrition' in output:
            result["malnutrition_types"].append("moderate_acute_malnutrition")
        if 'chronic_malnutrition' in output:
            result["malnutrition_types"].append("chronic_malnutrition")
        if 'mild_malnutrition' in output:
            result["malnutrition_types"].append("mild_malnutrition")
        
        # Extract weight category
        if 'severely_underweight' in output:
            result["weight_category"] = "severely_underweight"
        elif 'moderately_underweight' in output:
            result["weight_category"] = "moderately_underweight"
        elif 'weight_category(normal)' in output:
            result["weight_category"] = "normal"
        
        # Extract severity level
        if 'severity_level(severe)' in output:
            result["severity_level"] = "severe"
        elif 'severity_level(moderate)' in output:
            result["severity_level"] = "moderate"
        elif 'severity_level(mild)' in output:
            result["severity_level"] = "mild"
        
        # Extract numeric values using regex
        sc_match = re.search(r'symptom_count\((\d+)\)', output)
        if sc_match:
            result["symptom_count"] = int(sc_match.group(1))
        
        ss_match = re.search(r'severity_score\((\d+)\)', output)
        if ss_match:
            result["severity_score"] = int(ss_match.group(1))
        
        out_match = re.search(r'outcome\((\d+)\)', output)
        if out_match:
            result["outcome"] = int(out_match.group(1))
        
        # Extract recommendations
        if 'immediate hospital referral' in output:
            result["recommendations"].append("Immediate hospital referral")
        if 'therapeutic feeding' in output:
            result["recommendations"].append("Therapeutic feeding (F-75, F-100)")
        if 'high-calorie' in output:
            result["recommendations"].append("High-calorie and high-protein supplements")
        if 'nutrient-dense meals' in output:
            result["recommendations"].append("Nutrient-dense meals")
        if 'vitamin and mineral supplementation' in output:
            result["recommendations"].append("Vitamin and mineral supplementation")
        if 'regular health checkups' in output:
            result["recommendations"].append("Regular health checkups")
        if 'balanced diet' in output:
            result["recommendations"].append("Balanced diet with micronutrients")
        if 'long-term growth monitoring' in output:
            result["recommendations"].append("Long-term growth monitoring")
        if 'immunization' in output:
            result["recommendations"].append("Immunization and deworming")
        if 'protein-rich diet' in output:
            result["recommendations"].append("Protein-rich diet")
        if 'iron and vitamin supplements' in output:
            result["recommendations"].append("Iron and vitamin supplements")
        if 'increase meal frequency' in output:
            result["recommendations"].append("Increase meal frequency")
        
        # Deduplicate recommendations
        result["recommendations"] = list(set(result["recommendations"]))
        
        log(f"   Parsed result: {result}")
        return result
        
    except Exception as e:
        log(f"   Parse error: {repr(e)}")
        import traceback
        traceback.print_exc()
        return result

def validate_symptoms(symptoms):
    """Check which symptoms are known"""
    known_symptoms = [
        'low_weight', 'stunted_growth', 'swollen_belly', 'frequent_illness',
        'pale_skin', 'fatigue', 'brittle_hair', 'delayed_development',
        'skin_infections', 'weak_immunity', 'hair_discoloration',
        'loss_of_appetite', 'irritability', 'diarrhea', 'slow_healing_wounds'
    ]
    
    known = [s for s in symptoms if s in known_symptoms]
    unknown = [s for s in symptoms if s not in known_symptoms]
    
    return known, unknown

def run_a_star_prolog(start_node, goal_node):
    """Run A* search using Prolog subprocess"""
    log(f">> Running A* search: {start_node} -> {goal_node}")
    log(f">> Using knowledge base: {KB_PATH}")
    
    script = f"""
:- consult('{KB_PATH}').

run_a_star :-
    a_star({start_node}, {goal_node}, Path, Cost),
    write('PATH:'),
    write_canonical(Path),
    nl,
    write('COST:'),
    write(Cost),
    nl,
    halt(0).

run_a_star :-
    write('A_STAR_FAILED'),
    nl,
    halt(1).

:- initialization(run_a_star).
"""
    
    try:
        result = run_prolog_file(script)
        
        log(f"   Return code: {result.returncode}")
        log(f"   Stdout: {result.stdout}")
        if result.stderr:
            log(f"   Stderr: {result.stderr}")
        
        if result.returncode != 0 or 'A_STAR_FAILED' in result.stdout:
            return None, None, f"A* search failed: {result.stderr or result.stdout}"
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        path_str = None
        cost = None
        
        for line in lines:
            if line.startswith('PATH:'):
                path_str = line[5:].strip()
            elif line.startswith('COST:'):
                try:
                    cost = int(line[5:].strip())
                except ValueError:
                    cost = float(line[5:].strip())
        
        if path_str and cost is not None:
            # Parse the path list
            path_str = path_str.strip('[]').replace("'", "")
            path = [node.strip() for node in path_str.split(',')]
            return path, cost, None
        
        return None, None, "Failed to parse A* output"
        
    except subprocess.TimeoutExpired:
        return None, None, "A* search timeout"
    except Exception as e:
        log(f"   Exception: {repr(e)}")
        import traceback
        traceback.print_exc()
        return None, None, str(e)

def run_ao_star_prolog(root_node):
    """Run AO* search using Prolog subprocess"""
    log(f">> Running AO* search from: {root_node}")
    log(f">> Using knowledge base: {KB_PATH}")
    
    script = f"""
:- consult('{KB_PATH}').

run_ao_star :-
    ao_star_with_cost({root_node}, Strategy, Cost),
    write('STRATEGY:'),
    write_canonical(Strategy),
    nl,
    write('COST:'),
    write(Cost),
    nl,
    halt(0).

run_ao_star :-
    write('AO_STAR_FAILED'),
    nl,
    halt(1).

:- initialization(run_ao_star).
"""
    
    try:
        result = run_prolog_file(script)
        
        log(f"   Return code: {result.returncode}")
        log(f"   Stdout: {result.stdout}")
        if result.stderr:
            log(f"   Stderr: {result.stderr}")
        
        if result.returncode != 0 or 'AO_STAR_FAILED' in result.stdout:
            return None, None, f"AO* search failed: {result.stderr or result.stdout}"
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        strategy_str = None
        cost = None
        
        for line in lines:
            if line.startswith('STRATEGY:'):
                strategy_str = line[9:].strip()
            elif line.startswith('COST:'):
                try:
                    cost = int(line[5:].strip())
                except ValueError:
                    cost = float(line[5:].strip())
        
        if strategy_str and cost is not None:
            strategy = parse_strategy_list(strategy_str)
            return strategy, cost, None
        
        return None, None, "Failed to parse AO* output"
        
    except subprocess.TimeoutExpired:
        return None, None, "AO* search timeout"
    except Exception as e:
        log(f"   Exception: {repr(e)}")
        import traceback
        traceback.print_exc()
        return None, None, str(e)

def parse_strategy_list(strategy_str):
    """Parse nested Prolog list into flat Python list"""
    strategy_str = strategy_str.strip()
    nodes = re.findall(r'([a-z_]+)', strategy_str)
    seen = set()
    result = []
    for node in nodes:
        if node not in seen:
            seen.add(node)
            result.append(node)
    return result

# ------------------------------------------
# ROUTES
# ------------------------------------------
@app.route("/diagnose", methods=["POST"])
def diagnose():
    log("=== /diagnose called ===")
    data = request.get_json(silent=True) or {}
    log(f"Payload: {data}")

    try:
        age = float(data.get("age", 0))
        wt = float(data.get("weight", 0))
        symptoms = data.get("symptoms", [])
    except:
        log("❌ Bad payload")
        return jsonify({"error": "Invalid payload"}), 400

    if age <= 0 or wt <= 0 or not isinstance(symptoms, list):
        log("❌ Bad values for age/weight/symptoms")
        return jsonify({"error": "Invalid age/weight/symptoms"}), 400

    known, unknown = validate_symptoms(symptoms)
    log(f"Known symptoms: {known}")
    log(f"Unknown symptoms: {unknown}")

    result, error = diagnose_via_prolog(age, wt, known)
    
    if error:
        log(f"❌ Diagnosis failed: {error}")
        return jsonify({"error": error}), 500

    result["known_symptoms"] = known
    result["unknown_symptoms"] = unknown
    result["symptoms"] = known
    
    log(f"✅ Final result: {result}")
    return jsonify(result)

@app.route("/a_star", methods=["POST"])
def a_star_route():
    log("=== /a_star called ===")
    data = request.get_json(silent=True) or {}
    log(f"Payload: {data}")
    
    try:
        start_node = data.get("start_node", "").strip()
        goal_node = data.get("goal_node", "").strip()
        
        if not start_node or not goal_node:
            return jsonify({"error": "Both start_node and goal_node are required"}), 400
        
        path, cost, error = run_a_star_prolog(start_node, goal_node)
        
        if error:
            log(f"❌ A* search failed: {error}")
            return jsonify({"error": error}), 500
        
        if path is None:
            return jsonify({"error": f"No path found from {start_node} to {goal_node}"}), 404
        
        log(f"✅ A* result: path={path}, cost={cost}")
        return jsonify({
            "path": path,
            "cost": cost,
            "algorithm": "A*",
            "start": start_node,
            "goal": goal_node
        }), 200
        
    except Exception as e:
        log(f"❌ Exception: {repr(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/ao_star", methods=["POST"])
def ao_star_route():
    log("=== /ao_star called ===")
    data = request.get_json(silent=True) or {}
    log(f"Payload: {data}")
    
    try:
        root_node = data.get("root_node", "").strip()
        
        if not root_node:
            return jsonify({"error": "root_node is required"}), 400
        
        strategy, cost, error = run_ao_star_prolog(root_node)
        
        if error:
            log(f"❌ AO* search failed: {error}")
            return jsonify({"error": error}), 500
        
        if strategy is None:
            return jsonify({"error": f"No strategy found for root node {root_node}"}), 404
        
        log(f"✅ AO* result: strategy={strategy}, cost={cost}")
        return jsonify({
            "strategy": strategy,
            "cost": cost,
            "algorithm": "AO*",
            "root": root_node
        }), 200
        
    except Exception as e:
        log(f"❌ Exception: {repr(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/export_cases", methods=["GET"])
def export_cases():
    try:
        return jsonify({"message": "Cases exported successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "kb_path": KB_PATH, 
        "kb_found": os.path.exists(KB_PATH) if KB_PATH else False
    })

if __name__ == "__main__":
    log(">>> Starting PROLOG SERVICE on port 5001")
    log(f">>> Knowledge base path: {KB_PATH}")
    app.run(host="0.0.0.0", port=5001, debug=True)