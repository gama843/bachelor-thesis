import os
import base64
from collections import defaultdict
import matplotlib.pyplot as plt
from openai import OpenAI
from data_generator import DataGenerator
from dataset_builder import DatasetBuilder
from utils import get_question_type_and_subtype, log_and_print

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

def evaluate_model(image_path, question, question_vector, expected_answer, answer_set, model_name="gpt-4o"):
    """
    Evaluate the model by sending an image, a question, and an answer set, and receive an answer.

    Parameters:
    - image_path (str): The file path to the image.
    - question (str): A single question about the image.
    - question_vector (list): A 11-bit binary vector representing the question.
    - expected_answer (str): The ground truth answer.
    - answer_set (list): A list of possible answers.
    - model_name (str): Name of the model to use (default: 'gpt-4o').

    Returns:
    - dict: A dictionary with question, expected_answer, model_answer, question_type, and question_subtype.
    """
    base64_image = encode_image(image_path)

    answer_options = "\n".join([f"- {answer}" for answer in answer_set])
    prompt = f"""
    Based on the provided image and question, select the most appropriate answer from the following options:
    {answer_options}

    Question: {question}

    Your reply must contain only one option from the list and nothing else.
    """

    payload = {
        "model": model_name,
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
        ]
    }

    try:
        response = client.chat.completions.create(**payload)
        model_answer = response.choices[0].message.content.strip()
    except Exception as e:
        model_answer = f"An error occurred: {str(e)}"

    question_type, question_subtype = get_question_type_and_subtype(question_vector)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "model_answer": model_answer,
        "question_type": question_type,
        "question_subtype": question_subtype
    }

def compute_accuracy(results):
    """
    Compute the accuracy of predictions against the ground truth answers.

    Parameters:
    - results (list of dicts): Each dict contains 'expected_answer' and 'model_answer'.

    Returns:
    - float: The accuracy as a percentage.
    """
    correct = sum(r["model_answer"] == r["expected_answer"] for r in results)
    return (correct / len(results)) if results else 0.0

def breakdown_by(results, key):
    """
    Group results by a specified key and compute accuracy for each group.

    Parameters:
    - results (list of dict): A list of result dictionaries, each containing at least the given key and
      the fields 'expected_answer' and 'model_answer'.
    - key (str): The key in each result dict to group by (e.g., 'question_type' or 'question_subtype').

    Returns:
    - dict: A mapping from each unique value of the key to the corresponding accuracy (float).
    """
    buckets = defaultdict(list)
    for r in results:
        buckets[r[key]].append(r)
    return {k: compute_accuracy(v) for k, v in buckets.items()}

if __name__ == "__main__":
    img_dim = 75
    num_images = 25
    model_name = "o1"

    experiment_dir = "llmso1"
    data_dir = os.path.join(experiment_dir, 'data')
    os.makedirs(experiment_dir, exist_ok=True)

    generator = DataGenerator(data_dir)
    generator.generate_dataset(img_dim=img_dim, num_images=num_images)

    builder = DatasetBuilder(data_dir)
    answer_set = builder.answer_vocab

    results = []
    log_path = os.path.join(experiment_dir, "log.txt")
    log_file = open(log_path, "w")

    log_and_print(f"Model: {model_name}", log_file)
    log_and_print(f'Test samples: {len(builder.test_samples)}', log_file)
    for img_path, question, answer, question_vector in builder.test_samples:
        result = evaluate_model(img_path, question, question_vector, str(answer), answer_set, model_name)
        results.append(result)
        outcome = "PASS" if result['expected_answer'] == result['model_answer'] else "FAIL"
        log_and_print(f"Image: {img_path}", log_file)
        log_and_print(f"Question: {result['question']}", log_file)
        log_and_print(f"Expected answer: {result['expected_answer']}", log_file)
        log_and_print(f"Model answer: {result['model_answer']}", log_file)
        log_and_print(f"Question type: {result['question_type']}", log_file)
        log_and_print(f"Question subtype: {result['question_subtype']}", log_file)
        log_and_print(f"Outcome: {outcome}\n", log_file)

    overall_accuracy = compute_accuracy(results)
    type_acc = breakdown_by(results, "question_type")
    subtype_acc = breakdown_by(results, "question_subtype")

    log_and_print(f"Overall accuracy: {overall_accuracy:.4f}", log_file)
    log_and_print("\nAccuracy breakdown per question type:", log_file)
    for qtype in sorted(type_acc):
        log_and_print(f"  {qtype}: {type_acc[qtype]:.4f}", log_file)

    log_and_print("\nAccuracy breakdown per question subtype:", log_file)

    relational_subtypes = {"closest", "count", "furthest"}
    grouped_subtypes = {
        "relational": sorted([s for s in subtype_acc if s in relational_subtypes]),
        "non-relational": sorted([s for s in subtype_acc if s not in relational_subtypes])
    }

    for qtype in ["relational", "non-relational"]:
        log_and_print(f"  {qtype}:", log_file)
        for subtype in grouped_subtypes[qtype]:
            log_and_print(f"    {subtype}: {subtype_acc[subtype]:.4f}", log_file)

    labels = ["overall"] + list(type_acc.keys()) + list(subtype_acc.keys())
    accuracies = [overall_accuracy] + [type_acc[k] for k in type_acc] + [subtype_acc[k] for k in subtype_acc]
    colors = ["tab:red"] + ["tab:orange"] * len(type_acc) + ["tab:blue" if label in relational_subtypes else "tab:green" for label in subtype_acc]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, accuracies, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Accuracy')
    ax.set_title(f'Performance overview ({model_name})')
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8)
    for i, v in enumerate(accuracies):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(experiment_dir, f"performance_overview_{model_name}.png")
    plt.savefig(plot_path)
    log_and_print(f"Plot saved to {plot_path}", log_file)
    log_file.close()