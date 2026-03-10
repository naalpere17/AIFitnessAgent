import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .calendar_helper import get_calendar_summary, generate_add_to_calendar_link
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class FitnessAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.ical_url = "https://calendar.google.com/calendar/ical/achang93%40ucsc.edu/private-8afb01038a9cfac469aafafe81ad8793/basic.ics"
        
        # print(f"--- Loading {model_id} (Quantized) ---")
        #q_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_compute_dtype=torch.bfloat16,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_use_double_quant=True
        #)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
        
        self.system_prompt = (
            "You are a supportive Virtual Workout Buddy. You must distinguish between 'Checking' and 'Booking'.\n\n"
            "### DECISION LOGIC:\n"
            "1. IF the user asks 'Am I free' or 'What is my schedule':\n"
            "   - Simply answer yes/no based on the provided schedule.\n"
            "   - DO NOT suggest a workout. DO NOT use the [BOOK] tag.\n\n"
            "2. IF the user says 'I want to workout', 'Schedule it', or 'Let's do [exercise]':\n"
            "   - Confirm the time and use the tag: [BOOK: ISO_ID | Focus]\n\n"
            "### RULES:\n"
            "- Never use the [BOOK] tag for simple questions.\n"
            "- Never invent a workout focus if the user didn't ask for one.\n"
            "- If no workout is mentioned during a booking, use 'Workout' as the focus."
            # ... availability response examples included in system prompt logic
        )

    def save_summary_to_file(self, summary_text, filename="availability_summary.txt"):
        """Saves the calendar availability string to a .txt file."""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("--- CALENDAR AVAILABILITY EXPORT ---\n")
                f.write(summary_text)
            return f"✅ Availability exported to {filename}"
        except Exception as e:
            return f"❌ File Error: {e}"

    def generate_response(self, user_input):
        context = user_input
        if any(word in user_input.lower() for word in ["schedule", "free", "when", "time", "tomorrow"]):
            print("--- Checking calendar ---")
            availability = get_calendar_summary(self.ical_url)
            
            # Export to file immediately for other LLMs or systems to read
            print(self.save_summary_to_file(availability))
            
            context = f"User Schedule:\n{availability}\n\nUser Question: {user_input}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

        # PARSING LOGIC for Booking Tags
        match = re.search(r'\[BOOK:\s*(.+?)\s*\|\s*(.+?)\]', response, re.DOTALL)
        if match:
            iso_id, focus = match.group(1).strip(), match.group(2).strip()
            calendar_link = generate_add_to_calendar_link(iso_id, focus)
            response = re.sub(r'\s*\[BOOK:.*?\]', '', response, flags=re.DOTALL).strip()
            response += f"\n\n📅 **Invite Ready!** [Click here to add to your calendar]({calendar_link})"

        return response

if __name__ == "__main__":
    agent = FitnessAgent()
    while True:
        ui = input("You: ")
        if ui.lower() in ["exit", "quit"]: break
        print(f"\nAgent: {agent.generate_response(ui)}\n")
