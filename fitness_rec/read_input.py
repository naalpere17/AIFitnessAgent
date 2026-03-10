import json
import re
from typing import List, Optional
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, ValidationError


# ============================================================
# 1️⃣ ENUMS
# ============================================================

class GoalEnum(str, Enum):
    weight_loss = "weight_loss"
    muscle_gain = "muscle_gain"
    general_fitness = "general_fitness"


class IntensityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# ============================================================
# 2️⃣ PYDANTIC SCHEMA
# ============================================================

class WorkoutIntake(BaseModel):
    goal: Optional[GoalEnum] = None
    intensity: Optional[IntensityEnum] = None
    equipment: List[str] = []
    injuries: List[str] = []
    frequency: Optional[int] = None


# ============================================================
# 3️⃣ LOAD HUGGING FACE MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


# ============================================================
# 4️⃣ SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a strict workout data extraction engine.

Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

Mappings:
- "get shredded", "lose fat", "cut" → weight_loss
- "build muscle", "bulk" → muscle_gain
- "stay fit", "healthy" → general_fitness

Intensity:
- beginner → low
- moderate → medium
- intense → high

Schema:
{
  "goal": "weight_loss | muscle_gain | general_fitness | null",
  "intensity": "low | medium | high | null",
  "equipment": ["string"],
  "injuries": ["string"],
  "frequency": integer | null
}
"""


# ============================================================
# 5️⃣ JSON CLEANING
# ============================================================

def clean_json_output(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"```(json)?", "", text)
        text = text.replace("```", "").strip()

    return text


# ============================================================
# 6️⃣ GENERATION FUNCTION
# ============================================================

def generate_response(user_input: str):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            top_p=0.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt portion
    response = response[len(text):]

    return clean_json_output(response)


# ============================================================
# 7️⃣ MAIN PROCESSOR
# ============================================================

def process_user_request(user_text: str):

    raw_output = generate_response(user_text)

    try:
        parsed = json.loads(raw_output)
        validated = WorkoutIntake(**parsed)

        print("\n✅ Saved to DB:")
        print(validated.model_dump())

        return validated.model_dump()

    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw": raw_output}

    except ValidationError as e:
        return {"error": "Schema validation failed", "details": e.errors()}


# ============================================================
# 8️⃣ TEST
# ============================================================

if __name__ == "__main__":

    user_input = (
        "I want to get shredded for summer. "
        "I only have dumbbells and a pull-up bar, "
        "and I can work out 4 times a week. "
        "My left knee is weak."
    )

    result = process_user_request(user_input)

    print("\n📦 Final Output:")
    print(json.dumps(result, indent=2))