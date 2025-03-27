import os
from openai import OpenAI
import base64
from data_generator import DataGenerator
from dataset_builder import DatasetBuilder

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=openai_api_key)

def encode_image(image_path):
    """
    Encode an image to a Base64 string.

    Parameters:
    - image_path (str): The file path to the image.

    Returns:
    - str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_model(image_path, qa_pairs, answer_set):
    """
    Evaluate the model by sending an image, answer set and multiple questions, and receive answers.

    Parameters:
    - image_path (str): The file path to the image.
    - qa_pairs (list of dicts): A list of dictionaries, each containing a 'question' and its corresponding 'answer'.
    - answer_set (list): A list of possible answers.

    Returns:
    - list of dicts: Each dictionary contains the 'question', 'expected_answer', and 'model_answer'.
    """

    base64_image = encode_image(image_path)

    results = []

    for qa in qa_pairs:
        question = qa['question']
        expected_answer = qa['answer']

        answer_options = "\n".join([f"- {answer}" for answer in answer_set])
        prompt = f"""
        Based on the provided image and question, select the most appropriate answer from the following options:
        {answer_options}

        Question: {question}

        Your reply must contain only one option from the list and nothing else.
        """

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        try:
            response = client.chat.completions.create(**payload)
            model_answer = response.choices[0].message.content.strip()
        except Exception as e:
            model_answer = f"An error occurred: {str(e)}"

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer
        })

    return results

if __name__ == "__main__":
    image_path = 'experiments/27032025_10000_relational_cnn_binary/data/1.png'
    qa_pairs = [
        {"question": "What is the shape of the object that is farthest from the orange object?", "answer": "circle"},
        {"question": "What is the color of the object that is closest to the blue object?", "answer": "orange"},
        {"question": "How many objects have the shape of the yellow object?", "answer": "2"},
        {"question": "Is the blue object on the top or the bottom?", "answer": "top"},
        {"question": "What is the shape of the blue object?", "answer": "square"},
        {"question": "Is the red object on the left or the right?", "answer": "right"}
    ]
    answer_set = ['square', 'circle', 'bottom', 'right', 'top', 'left', '4', '3', 'yellow', 'green', 'orange', 'red', 'blue', 'pink', '2', '5', '6', '1', '<UNK>']

    results = evaluate_model(image_path, qa_pairs, answer_set)
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Expected Answer: {result['expected_answer']}")
        print(f"Model Answer: {result['model_answer']}")
        print()