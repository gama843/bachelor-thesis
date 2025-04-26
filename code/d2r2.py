import argparse
import torch
import numpy as np
import random
import os
import warnings

warnings.filterwarnings("ignore")

# reproducibility: 

# fix the random seed 
seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# force deterministic torch/CUDA ops
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn    

from data_generator import DataGenerator
from models import ModelConstructor
from dataset_builder import DatasetBuilder, collate_fn
from training import validate_one_epoch, train_and_validate, save_train_answer_distribution, compute_baseline_performance
from llms import run_llm_evaluation
from utils import get_experiment_name

def main():
    parser = argparse.ArgumentParser(
        description=("The software provides several functions which together form a general framework for testing "
                     "relational reasoning capabilities of neural networks. This implementation offers a suite of 2D "
                     "perception tasks. More tasks can be easily added later. You can use it to parse your own dataset "
                     "and then test a custom or a provided neural network and it will generate a detailed performance "
                     "report. You can run it interactively or use it in a script of your own.")
    )
    
    parser.add_argument('-g2', '--generate2D', type=str, help='Generate a new 2D dataset in the provided path.')
    parser.add_argument('-e', '--eval', action='store_true', help='Run evaluation on the given model and dataset.')
    parser.add_argument('-m', '--model_path', type=str, help='Path to the model to be evaluated.')
    parser.add_argument('-d', '--dataset_path', type=str, help='Path to the dataset for evaluation.')
    parser.add_argument('-r', '--report_path', type=str, help='Path to save the evaluation report.')
    parser.add_argument('-l', '--llm_eval', type=str, help='Path to the experiment folder.')
    parser.add_argument('-n', '--llm_name', type=str, help='The LLM you want to test, valid options are: "gpt-4o", "gpt-4.5-preview", "o1", "all". Default is "all".')

    args = parser.parse_args()

    if not any(vars(args).values()):
        # if no arguments are passed, run the default behavior (dataset generation and model training)
        print("No arguments provided. Running default dataset generation and model training.")
        
        img_dim = 75
        num_images = 10000
        num_epochs = 25
        batch_size = 64
        model_type = 'relational'
        image_form = 'image'
        img_arch = 'cnn'
        question_form = 'string'
        note = 'final'

        experiment_dir = get_experiment_name(num_images, model_type, image_form, question_form, seed, img_arch, note)
        data_dir = os.path.join(experiment_dir, 'data')

        generator = DataGenerator(data_dir)
        generator.generate_dataset(img_dim=img_dim, num_images=num_images)

        builder = DatasetBuilder(data_dir)
        builder.save()
        print('Dataset builder sucessfully saved.')
        
        save_train_answer_distribution(experiment_dir, builder)

        train_loader = DataLoader(builder.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(builder.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        test_loader = DataLoader(builder.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        compute_baseline_performance(test_loader, builder.answer_vocab, experiment_dir)

        model_constructor = ModelConstructor()
        model = model_constructor.load_model(
            img_arch=img_arch,
            model_type=model_type,
            vocab_size=len(builder.vocab),
            embed_size=32,
            hidden_size=128,
            num_layers=1,
            num_classes=len(builder.answer_vocab),
            question_form=question_form,
            image_form=image_form
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, experiment_dir, question_form, image_form)
    
    else:
        # handle the provided arguments logic
        if args.llm_eval:
            if args.llm_name:
                run_llm_evaluation(args.llm_eval, args.llm_name)
            else:
                run_llm_evaluation(args.llm_eval)

        if args.generate2D:
            generator = DataGenerator(args.generate2D)
            generator.generate_dataset(img_dim=224, num_images=10)
            print(f"Dataset generated at {args.generate2D}")

        if args.eval:
            if not args.model_path or not args.dataset_path or not args.report_path:
                raise ValueError("For evaluation, you must provide --model_path, --dataset_path, and --report_path.")
            
            print(f"Loading dataset from {args.dataset_path}")
            builder = DatasetBuilder(args.dataset_path)
            test_loader = DataLoader(builder.test_dataset, batch_size=1, shuffle=False)

            print(f"Loading model from {args.model_path}")
            model_constructor = ModelConstructor()
            model = model_constructor.load_model(
                model_type='baseline',  # assuming baseline model
                vocab_size=len(builder.vocab),
                embed_size=128,
                hidden_size=256,
                num_layers=1,
                feature_dim=512,
                g_theta_dim=256,
                f_phi_dim=128,
                num_classes=len(builder.answer_vocab)
            )
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            # load model weights
            if os.path.exists(args.model_path):
                model.load_state_dict(torch.load(args.model_path))
                print(f"Model loaded successfully from {args.model_path}")
            else:
                raise FileNotFoundError(f"Model file {args.model_path} not found.")

            # evaluation
            print("Starting evaluation...")
            criterion = nn.CrossEntropyLoss()
            test_loss, test_accuracy = validate_one_epoch(model, test_loader, criterion, device)
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

            # save evaluation report
            with open(args.report_path, 'w') as report_file:
                report_file.write(f'Test Loss: {test_loss:.4f}\n')
                report_file.write(f'Test Accuracy: {test_accuracy:.4f}\n')
            print(f"Evaluation report saved at {args.report_path}")


if __name__ == "__main__":
    main()