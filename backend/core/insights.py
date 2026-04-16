import os
import json
from dotenv import load_dotenv

load_dotenv()
_AI_DISABLED_REASON = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def get_openai_client():
    global _AI_DISABLED_REASON
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        _AI_DISABLED_REASON = "missing_openai_api_key"
        return None
    if OpenAI is None:
        _AI_DISABLED_REASON = "openai_package_missing"
        return None
    return OpenAI(api_key=api_key)


def get_model_category(model_name: str) -> str:
    """Categorize model for factual AI explanations."""
    name = model_name.lower()
    if "logistic" in name or "linear" in name:
        return "Linear Model (captures direct, straight-line relationships)"
    if "forest" in name or "xgboost" in name or "gradient" in name or "tree" in name:
        return "Tree-based Ensemble (partitions data into non-linear branches)"
    if "svm" in name or "svc" in name or "svr" in name:
        return "Kernel-based Model (finds complex boundaries in high dimensions)"
    return "Machine Learning Model"


def generate_insights(profile: dict, results: dict) -> dict:
    fallback = _fallback_generate_insights(profile, results)
    client = get_openai_client()
    if not client:
        fallback["ai_disabled_reason"] = _AI_DISABLED_REASON or "ai_unavailable"
        return fallback

    best_model = results.get('best_model', 'Unknown')
    category = get_model_category(best_model)
    score = results.get('score', 0)
    is_low_perf = score < 50
    perf_context = "CRITICAL: The model performance is very low (<50%). Focus on explaining that the features might not be strong enough or the data is noisy, rather than praising the model." if is_low_perf else ""

    prompt = f"""
    You are an expert Data Science AI Coach. The user ran an AutoML experiment.
    Dataset Profile: {json.dumps(profile)}
    Model Results: {json.dumps(results)}
    Winner Model: {best_model} ({category})
    
    {perf_context}

    Based on the dataset health and the model results, do three things:
    1. Give an "Explain Like I'm 5" (eli5) explanation of why this specific model type ({category}) won and how it generally works. 
       - IMPORTANT: Do NOT call a Linear Model "non-linear".
       - If performance is low, be honest about why.
    2. Give 1 to 3 "coach_msgs", which are brief, actionable tips on data preprocessing.
    3. Give 2 to 3 "why_won" bullet points explaining the technical reason for this model's success (e.g., "Captures non-linear branches", "Handles high-dimensional signals", "Robust to small data noise").

    Respond ONLY with a valid JSON strictly following this structure:
    {{
      "eli5": "your explain-like-i'm-5 string",
      "coach": ["tip 1", "tip 2"],
      "why_won": ["reason 1", "reason 2"]
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=600,
            messages=[
                {"role": "system", "content": "You are a JSON-only data science assistant. Output only valid JSON. Do not use the word 'Accuracy' for regression tasks."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        data = json.loads(content)
        return {
            "eli5": data.get("eli5", fallback["eli5"]), 
            "coach": data.get("coach", fallback["coach"]),
            "why_won": data.get("why_won", fallback["why_won"])
        }
    except Exception as e:
        msg = str(e).lower()
        if "invalid_api_key" in msg or "incorrect api key" in msg or "401" in msg:
            fallback["ai_disabled_reason"] = "invalid_openai_api_key"
        else:
            fallback["ai_disabled_reason"] = "openai_api_error"
        return fallback


def generate_story(profile: dict, results: dict) -> str:
    fallback = _fallback_generate_story(profile, results)
    client = get_openai_client()
    if not client:
        return fallback

    best_model = results.get('best_model', 'RandomForest')
    category = get_model_category(best_model)
    score = results.get('score', 0)
    is_low_perf = score < 50
    cols = profile.get('cols', 0)
    
    perf_msg = "but note that overall performance is low, suggesting the dataset features may be weak." if is_low_perf else ""
    
    # Specific "Why Linear Won" logic
    is_linear = "Linear" in category
    why_linear_won = ""
    if is_linear and cols <= 3:
        why_linear_won = """
        Why Linear Regression won:
        - Few features (only {cols} columns)
        - Strong linear relationship detected between features and target
        - Low noise in the data allows a simple model to generalize best
        """

    prompt = f"""
    Write a 3-4 sentence fun, beginner-friendly "Wrap Up Story" analyzing an AutoML job.
    Make it engaging, using emojis if appropriate.
    Dataset: {profile.get('rows', 0)} rows, {cols} columns.
    Winner Model: {best_model} ({category})
    Metric: {results.get('metric_name', 'Score')} = {score}%
    
    {why_linear_won}
    
    Story should explain what the dataset looked like, which algorithm dominated,
    and broadly why it succeeded based on its category ({category}).
    {perf_msg}
    
    FACTUAL RULES:
    - Linear/Logistic Regression are LINEAR models (straight-line logic).
    - Random Forest/XGBoost are TREE-BASED ensembles.
    - If Linear model won, explain that the data shows clear direct relationships.
    - NEVER use the word 'Accuracy' for Regression. Use 'R² Score' instead.
    - If accuracy/R² is low, mention the data was noisy or complex!
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return fallback


def _fallback_generate_insights(profile: dict, results: dict) -> dict:
    coach_msgs = []
    if profile.get('imbalance') == "High ⚠️":
        coach_msgs.append(
            "Dataset is highly imbalanced → try applying SMOTE or adjusting "
            "class weights for better minority class recall."
        )

    if profile.get('missing_pct', 0) > 5.0:
        coach_msgs.append(
            f"We noticed {profile.get('missing_pct')}% missing values. "
            "We used median imputation, but you might want domain-specific fills."
        )

    if not coach_msgs:
        coach_msgs.append(
            "Your dataset health is excellent! The automated preprocessing handled everything perfectly."
        )

    best_model = results.get('best_model', 'RandomForest')
    score = results.get('score', 0)
    category = get_model_category(best_model)
    cols = profile.get('cols', 0)
    metric = results.get('metric_name', 'Score')
    
    is_r2 = "R²" in metric
    display_score = f"{score/100:.3f}" if is_r2 else f"{score}%"
    
    # Specific "Why Model Won" logic for fallbacks
    why_won = []
    if "Linear" in category and cols <= 3:
        why_won = [
            f"Only {cols} features → allows for simple, direct relationships",
            "Strong linear correlation detected between features and target",
            "Low noise in data favors high-bias linear models over complex trees"
        ]
    elif "Tree" in category:
        why_won = [
            "Captures non-linear decision boundaries",
            "Handles complex feature interactions that a linear model misses",
            "Robust to outliers thanks to recursive partitioning"
        ]
    else:
        why_won = ["Optimized architecture for this specific dataset structure"]

    if score < 50:
        explainer = (
            f"Your data was quite challenging to predict! We tried several models, and "
            f"{best_model} ({category}) was the most promising, but it only reached {display_score} {metric}. "
            f"This often means the features in your dataset don't have a strong enough "
            f"connection to what you're trying to predict yet."
        )
    else:
        if "Linear" in category:
            complexity_part = f" With only {cols} features, the relationship is simpler and more direct." if cols <= 3 else ""
            explainer = (
                f"Your data shows clear, direct relationships! That's why a Linear Model like "
                f"{best_model} worked so well — it found the best straight-line path to "
                f"connect your features to the target.{complexity_part}"
            )
        else:
            explainer = (
                f"Your data has complex patterns that don't follow a straight line. "
                f"That's why a Tree-based model like {best_model} performed best — it "
                f"breaks the data down into many small decision branches to find the answer."
            )

    return {"eli5": explainer, "coach": coach_msgs, "why_won": why_won}


def _fallback_generate_story(profile: dict, results: dict) -> str:
    rows = profile.get('rows', 0)
    cols = profile.get('cols', 0)
    best_model = results.get('best_model', 'RandomForest')
    score = results.get('score', 0)
    metric = results.get('metric_name', 'Score')
    category = get_model_category(best_model)
    
    is_r2 = "R²" in metric
    display_score = f"{score/100:.3f}" if is_r2 else f"{score}%"
    
    reason = "it found strong linear relationships in your features" if "Linear" in category else "its ability to capture complex non-linear patterns"

    story = (
        f"We analyzed your dataset containing {rows} rows and {cols} columns, "
        f"and automatically tested top ML models. "
        f"**{best_model}** performed best because {reason}. "
        f"The model achieved an {metric} of **{display_score}**, indicating {'excellent fit' if score > 80 else 'a good start'}! "
        f"All preprocessing (imputation, scaling, encoding) was automatically applied."
    )
    return story


def chat_with_model(prompt: str, context: dict) -> str:
    client = get_openai_client()
    if not client:
        if "feature" in prompt.lower():
            return (
                "Based on SHAP analysis, the top features are driving the majority of my decisions. "
                "Check the SHAP chart in the Deep Analysis tab for details."
            )
        elif "why" in prompt.lower() or "prediction" in prompt.lower():
            return (
                "I made that prediction based on patterns in the training data. "
                "The SHAP tab shows exactly which features pushed the decision."
            )
        else:
            return (
                "I primarily look for complex non-linear splits in the data. "
                "Try asking what features matter most, or why a specific prediction was made. "
                "(Tip: Set OPENAI_API_KEY for full AI-powered answers.)"
            )

    system_prompt = (
        f"You are the AI representing an AutoML trained model. "
        f"Context of the model and dataset: {json.dumps(context)}."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception:
        return "AI assistant is currently unavailable. Check OPENAI_API_KEY; core training and prediction are unaffected."


# ── Feature 9: Natural Language → ML Intent Parser ───────────────────────────

def parse_nl_intent(prompt: str, profile: dict = None) -> dict:
    """
    Parse a natural language prompt into a structured ML training config.

    Returns:
        {
            task: "train" | "predict" | "explore",
            target: suggested target column name or None,
            goal: "Accuracy" | "Speed" | "Explainability",
            mode: "Fast" | "Balanced" | "Full",
            task_type: "classification" | "regression" | None,
            confidence: 0-100,
            explanation: human-readable parse summary,
            ready_to_train: bool,
        }
    """
    profile = profile or {}
    prompt_lower = prompt.lower().strip()

    # ── Try Gemini if key available ───────────────────────────────────────────
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        result = _gemini_parse_intent(prompt, profile, gemini_key)
        if result:
            return result

    # ── Rule-based keyword fallback ───────────────────────────────────────────
    return _rule_based_parse_intent(prompt_lower, profile)


def _gemini_parse_intent(prompt: str, profile: dict, api_key: str) -> dict | None:
    """Attempt Gemini-powered NL parse. Returns None on failure."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        col_names = list(profile.get("column_stats", {}).keys())[:20]
        system = f"""You are an AutoML assistant. Parse the user's request into a JSON config.
Available columns: {col_names}

Return ONLY valid JSON:
{{
  "task": "train",
  "target": "<column name or null>",
  "goal": "Accuracy",
  "mode": "Balanced",
  "task_type": "classification or regression or null",
  "confidence": 85,
  "explanation": "one sentence summary",
  "ready_to_train": true
}}

Rules:
- goal options: "Accuracy", "Speed", "Explainability"
- mode options: "Fast", "Balanced", "Full"
- If user says "quick", "fast", "speed" → mode=Fast
- If user says "best", "optimize" → mode=Full
- If user says "explain", "understand" → goal=Explainability
- For regression targets (price, salary, age, revenue, score) → task_type=regression
"""
        response = model.generate_content(f"{system}\n\nUser: {prompt}")
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception:
        return None


def _rule_based_parse_intent(prompt_lower: str, profile: dict) -> dict:
    """Keyword-based fallback parser."""

    # Task type detection
    task = "train"
    if any(w in prompt_lower for w in ["predict", "forecast", "estimate"]):
        task = "train"
    elif any(w in prompt_lower for w in ["explore", "analyze", "profile", "understand data"]):
        task = "explore"

    # Mode
    mode = "Balanced"
    if any(w in prompt_lower for w in ["quick", "fast", "rapid", "speed"]):
        mode = "Fast"
    elif any(w in prompt_lower for w in ["best", "optimal", "full", "maximum", "thorough"]):
        mode = "Full"

    # Goal
    goal = "Accuracy"
    if any(w in prompt_lower for w in ["explain", "interpret", "understand", "why", "shap"]):
        goal = "Explainability"
    elif any(w in prompt_lower for w in ["fast", "quick", "speed", "lightweight"]):
        goal = "Speed"

    # Target column suggestions from prompt keywords
    target = None
    regression_targets = ["price", "salary", "revenue", "sales", "cost", "income",
                          "age", "score", "rate", "value", "amount", "quantity"]
    classification_targets = ["churn", "fraud", "survival", "survived", "default",
                               "class", "label", "category", "diagnosis", "outcome",
                               "spam", "sentiment", "disease", "status"]

    inferred_task_type = None
    for word in prompt_lower.split():
        if any(rt in word for rt in regression_targets):
            inferred_task_type = "regression"
        if any(ct in word for ct in classification_targets):
            inferred_task_type = "classification"

    # Try to match against known columns
    col_names = list(profile.get("column_stats", {}).keys())
    for col in col_names:
        if col.lower() in prompt_lower:
            target = col
            break

    # If no column matched, suggest from domain keywords
    if not target:
        for col in col_names:
            col_l = col.lower()
            if any(rt in col_l for rt in regression_targets + classification_targets):
                target = col
                break
        if not target and col_names:
            target = col_names[-1]  # default to last column

    confidence = 60
    if target:
        confidence += 15
    if inferred_task_type:
        confidence += 10
    if mode != "Balanced":
        confidence += 5

    explanation_parts = [f"Detected task: {task}"]
    if target:
        explanation_parts.append(f"suggested target: '{target}'")
    if inferred_task_type:
        explanation_parts.append(f"problem type: {inferred_task_type}")
    explanation_parts.append(f"mode: {mode}, goal: {goal}")

    return {
        "task": task,
        "target": target,
        "goal": goal,
        "mode": mode,
        "task_type": inferred_task_type,
        "confidence": confidence,
        "explanation": " | ".join(explanation_parts),
        "ready_to_train": bool(target and task == "train"),
    }
