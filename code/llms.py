import os
import base64
from collections import defaultdict
from openai import OpenAI
from dataset_builder import DatasetBuilder
from utils import get_question_type_and_subtype, log_and_print, plot_accuracy_breakdown

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

def model_eval(experiment_dir, model_name):
    """
    Evaluate a language model's performance on a test set and log detailed results.

    Parameters:
    -----------
    experiment_dir : str
        Path to the experiment directory containing the dataset builder and where logs/plots will be saved.
    
    model_name : str
        Name of the language model to evaluate (e.g., 'gpt-4o', 'o1', 'gpt-4.5-preview').

    Description:
    ------------
    Loads the test samples from a saved DatasetBuilder, evaluates each sample using the specified model,
    logs each result to a file, computes accuracy overall and by question (sub)type, and saves a horizontal
    bar plot summarizing the performance metrics. Accuracy for 'relational' and 'non-relational' subtypes
    is grouped and color-coded in the plot.

    Output:
    -------
    - A detailed log file (log.txt) in the experiment directory.
    - A performance plot saved as performance_overview_<model_name>.png in the experiment directory.
    """
    builder = DatasetBuilder.load(os.path.join(experiment_dir, 'dataset_builder.pickle'))
    answer_set = builder.answer_vocab
    results = []
    log_path = os.path.join(experiment_dir, f"eval_log_{model_name}.txt")
    log_file = open(log_path, "w")

    log_and_print(f"Model: {model_name}", log_file)
    log_and_print(f'Test samples: {len(builder.test_samples)}', log_file)
    for img_path, question, answer, question_vector, _ in builder.test_samples:
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

    relational_subtypes = {"closest", "count", "farthest"}
    grouped_subtypes = {
        "relational": sorted([s for s in subtype_acc if s in relational_subtypes]),
        "non-relational": sorted([s for s in subtype_acc if s not in relational_subtypes])
    }

    for qtype in ["relational", "non-relational"]:
        log_and_print(f"  {qtype}:", log_file)
        for subtype in grouped_subtypes[qtype]:
            log_and_print(f"    {subtype}: {subtype_acc[subtype]:.4f}", log_file)

    plot_path = plot_accuracy_breakdown(
        overall_accuracy,
        type_acc,
        subtype_acc,
        model_name,
        experiment_dir
    )
    log_and_print(f"Plot saved to {plot_path}", log_file)
    log_file.close()

def run_llm_evaluation(experiment_dir, model_name='all'):
    """
    Run LLM evaluation on the specified model or all available models.

    Parameters:
    -----------
    experiment_dir : str
        Path to the experiment directory.
    model_name : str
        Name of the model to evaluate ('all', 'gpt-4o', 'o1', 'gpt-4.5-preview').

    Raises:
    -------
    ValueError
        If an unknown model_name is provided.
    """
    models = ['gpt-4o', 'o1', 'gpt-4.5-preview']
    model_name = model_name.lower()

    if model_name == 'all':
        for name in models:
            model_eval(experiment_dir, name)
    elif model_name in models:
        model_eval(experiment_dir, model_name)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Available options are: {models + ['all']}")