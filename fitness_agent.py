import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from calendar_helper import get_calendar_summary

class FitnessAgent:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"--- Loading {model_id} (Quantized) ---")
        
        # 1. Setup Quantization (Required for 4-bit loading)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # 2. Load Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 
        )

        # 3. Project Configuration
        # IMPORTANT: Replace this with your actual Google Calendar Secret iCal URL
        self.ical_url = "https://calendar.google.com/calendar/ical/achang93%40ucsc.edu/private-8afb01038a9cfac469aafafe81ad8793/basic.ics"
        
        # System Prompt derived from Silacci et al. (2026) goals
        self.system_message = (
            "You are a supportive Virtual Workout Buddy AI. Your goal is to provide social support, "
            "accountability, and adaptive coaching. You help users stay consistent with their goals. "
            "If a user mentions pain, fatigue, or injury, prioritize safety and recovery advice. "
            "When suggesting workout times, use the calendar data provided to be specific."
        )

    def generate_response(self, user_input):
        """
        Processes user input, checks if calendar data is needed, and generates a chat response.
        """
        context_query = user_input
        
        # Simple heuristic to trigger the calendar tool
        # In a more advanced version, the LLM would decide this itself.
        scheduling_keywords = ["schedule", "free time", "when can i", "find a slot", "workout window"]
        if any(word in user_input.lower() for word in scheduling_keywords):
            print("Agent logic: Fetching calendar gaps...")
            calendar_data = get_calendar_summary(self.ical_url)
            context_query = f"The user wants to find a time to work out. Here is their current schedule:\n{calendar_data}\n\nUser request: {user_input}"

        # Prepare Chat Template
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": context_query},
        ]

        # Tokenize and Generate
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.5,
                top_p=0.9
            )
        
        # Decode only the new text
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return response

# --- Main Interaction Loop ---
if __name__ == "__main__":
    try:
        agent = FitnessAgent()
        print("\n💪 Virtual Workout Buddy Online.")
        print("Type 'quit' to exit.\n")

        while True:
            user_text = input("You: ")
            if user_text.lower() in ["quit", "exit"]:
                print("Agent: Stay consistent! See you next time.")
                break

            reply = agent.generate_response(user_text)
            print(f"\nAgent: {reply}\n")

    except Exception as e:
        print(f"❌ Critical Error: {e}")
