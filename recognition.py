import ollama
import os

def identify_workout_machine(image_path, model_name="gemma3:27b"):
    # check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return None
    
    print(f'Identifying workout machine using image {model_name}...')

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                    'role': 'user',
                    'content': (
                        "Analyze this image and identify the workout/gym machine shown. "
                        "Provide only what the machine is called, without any additional information."
                    ),
                    'images': [image_path]
                },
            ],
        )
        content = response['message']['content'].strip()
        return content
    except ollama.ResponseError as e:
        print(f"Error: {e.error}")