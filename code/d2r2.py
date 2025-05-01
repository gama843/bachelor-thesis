import argparse
import sys
import os
import warnings
import random

import torch
import numpy as np

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

from argparse import RawDescriptionHelpFormatter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn    

from data_generator import DataGenerator
from models import ModelConstructor, BaselineModel, RelationalReasoningModel
from dataset_builder import DatasetBuilder, collate_fn
from training import validate_one_epoch, train_and_validate, save_train_answer_distribution, compute_baseline_performance
from lvlms import run_llm_evaluation
from utils import get_experiment_name

def main():
    parser = argparse.ArgumentParser(
        description=(
            "General framework for 2D relational‐reasoning experiments:\n"
            "  • Train/evaluate models\n"
            "  • Generate 2D D2R2 datasets\n"
            "  • Run LLM‐based evaluations"
        ),
        epilog=(
            "Basic usage examples:\n"
            "  Train & eval default model:\n"
            "    python d2r2.py --train [--img_dim 75 --num_images 10000 ...]\n\n"
            "  Generate a new dataset at this path (creates an experiment folder with data subfolder):\n"
            "    python d2r2.py --generate2D -x ./my_experiment_dir\n\n"
            "  Run model evaluation:\n"
            "    python d2r2.py --eval --model_path model.pth --experiment_dir exp_dir\n\n"
            "  Run LLM evaluation:\n"
            "    python d2r2.py --llm_eval exp_dir [--llm_name gpt-4o]\n"
        ),
        formatter_class=RawDescriptionHelpFormatter
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    parser.add_argument('-t', '--train', 
        action='store_true',
        help='Train and evaluate the default model.')

    parser.add_argument('-g2', '--generate2D',
        action='store_true', 
        help='Generate a new 2D dataset. Provide path through -x (experiment folder).')

    parser.add_argument('-l', '--llm_eval', 
        type=str, 
        help='Path to the experiment folder.')
    
    parser.add_argument('-e', '--eval', 
        action='store_true', 
        help='Run evaluation on the given model (-m) and dataset (-x).')

    parser.add_argument('-d', '--has_data', 
        action='store_true', 
        help='Use this flag if you have already generated your training data.')        

    parser.add_argument('-m', '--model_path', 
        type=str, 
        help='Path to the model to be evaluated.')
    parser.add_argument('-x', '--experiment_dir', 
        type=str, 
        help='Path to the experiment directory you want to evaluate or train a new model in.')
    parser.add_argument(
        '-n', '--llm_name',
        choices=['gpt-4o', 'gpt-4.5-preview', 'o1', 'all'],
        default='all',
        help='The LLM you want to test: gpt-4o, gpt-4.5-preview, o1, or all (default: all).')
    parser.add_argument('-q', '--question_form',
        choices=['binary', 'string'],
        default='binary',
        help='Form of the question: binary or string (default: binary).')
    parser.add_argument('-i', '--image_form',
        choices=['matrix', 'image'],
        default='matrix',
        help='Form of the image input: matrix or image (default: matrix).')
    parser.add_argument('--img_arch',
        type=str,
        default='cnn',
        help='Backbone for image encoder: "cnn" or "resnet" (default: "cnn").')
    parser.add_argument('--model_type',
        choices=['matrix', 'image'],
        default='baseline',
        help='Specifies the reasoning head used in the model (default: "baseline").')
    parser.add_argument('--embed_size',
        type=int,
        default=32,
        help='Dimensionality of the question input embedding (default: 32).')
    parser.add_argument('--batch_size',
        type=int,
        default=64,
        help='Batch size for training/evaluation.')
    parser.add_argument('--hidden_size',
        type=int,
        default=128,
        help='Hidden‐layer size for the reasoning modules (default: 128).')
    parser.add_argument('--num_layers',
        type=int,
        default=1,
        help='Number of layers in the LSTM question encoder (default: 1).')
    parser.add_argument('--img_dim',
        type=int,
        default=75,
        help='Square side length of the generated images (default: 75px).')
    parser.add_argument('--num_images',
        type=int,
        default=10000,
        help='Specifies how many images you want to generate in a new dataset instance.')
    parser.add_argument('--num_epochs',
        type=int,
        default=25,
        help='Specifies how many epochs you want to train for.')
    parser.add_argument('--lr',
        type=int,
        default=0.0001,
        help='Specifies learning rate.')    
    parser.add_argument('--note',
        type=str,
        default='',
        help='Note you want to append to your experiment name, empty by default.')

    args = parser.parse_args()

    if args.train:
        img_dim = args.img_dim
        num_images = args.num_images
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        model_type = args.model_type
        image_form = args.image_form
        img_arch = args.img_arch
        question_form = args.question_form
        note = args.note

        if not args.experiment_dir:
            experiment_dir = get_experiment_name(num_images, model_type, image_form, question_form, seed, img_arch, note)
        else:
            experiment_dir = args.experiment_dir
        
        data_dir = os.path.join(experiment_dir, 'data')

        if not args.has_data:
            generator = DataGenerator(data_dir)
            generator.generate_dataset(img_dim=img_dim, num_images=num_images)
            builder = DatasetBuilder(data_dir)
            builder.save()
            print('Dataset builder sucessfully saved.')
        else:
            builder = DatasetBuilder(data_dir)
            builder_path = os.path.join(experiment_dir, 'dataset_builder.pickle')
            builder.load(builder_path)
            print('Dataset builder sucessfully loaded.')
        
        save_train_answer_distribution(experiment_dir, builder)

        train_loader = DataLoader(builder.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(builder.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        test_loader = DataLoader(builder.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        compute_baseline_performance(test_loader, builder.answer_vocab, experiment_dir)

        model_constructor = ModelConstructor()
        model = model_constructor.load_model(
            img_arch=args.img_arch,
            model_type=args.model_type,
            vocab_size=len(builder.vocab),
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=len(builder.answer_vocab),
            question_form=args.question_form,
            image_form=args.image_form
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_and_validate(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, experiment_dir, question_form, image_form)
    
    elif args.llm_eval:
        if args.llm_name:
            run_llm_evaluation(args.llm_eval, args.llm_name)
        else:
            run_llm_evaluation(args.llm_eval)

    elif args.generate2D:
        experiment_dir = args.experiment_dir
        data_dir = os.path.join(experiment_dir, 'data')
        generator = DataGenerator(data_dir)
        generator.generate_dataset(img_dim=args.img_dim, num_images=args.num_images)
        print(f"Dataset generated at {data_dir}")

        builder = DatasetBuilder(data_dir)
        builder.save()
        print('Dataset builder sucessfully saved.')
        
        save_train_answer_distribution(experiment_dir, builder)

    elif args.eval:
        if not args.model_path or not args.experiment_dir:
            raise ValueError("For evaluation, you must provide --model_path and --experiment_dir.")

        data_path = os.path.join(args.experiment_dir, 'data')

        print("Starting evaluation...")
        print(f"Building test set from {data_path}...")
        builder = DatasetBuilder(data_dir=data_path)
        test_loader = DataLoader(
            builder.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        print("Computing baseline metrics...")
        save_train_answer_distribution(args.experiment_dir, builder)
        compute_baseline_performance(test_loader, builder.answer_vocab, args.experiment_dir)
        baseline_log = os.path.join(args.experiment_dir, 'baseline_performance.txt')
        baseline = None
        with open(baseline_log, 'r') as f:
            baseline = f.readlines()

        print(f"Loading model from {args.model_path}")
        model_constructor = ModelConstructor()
        model_class = None

        if args.model_type == 'baseline':
            model_class = BaselineModel(
                img_arch=args.img_arch,
                vocab_size=len(builder.vocab),
                embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_classes=len(builder.answer_vocab),
                question_form=args.question_form,
                image_form=args.image_form
            )
        elif args.model_type == 'relational':
            model_class = RelationalReasoningModel(
                img_arch=args.img_arch,
                vocab_size=len(builder.vocab),
                embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                num_classes=len(builder.answer_vocab),
                question_form=args.question_form,
                image_form=args.image_form
            )

        else:
            raise ValueError(f"Unknown model_type '{args.model_type}'. Valid options are 'baseline' or 'relational'.")
        
        model = model_constructor.load_model(
            model_type='custom',
            model_class=model_class,
            weights_path=args.model_path
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("Running model evaluation...")
        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy, test_breakdown, test_subtype_accuracy = validate_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            question_form=args.question_form,
            image_form=args.image_form
        )
        
        report_path = os.path.join(args.experiment_dir, 'eval_report.txt')

        with open(report_path, 'w') as f:
            f.write("=== Model Evaluation ===\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")

            f.write("=== Accuracy by Question Type ===\n")
            for category, acc in test_breakdown.items():
                f.write(f"{category}: {acc:.4f}\n")
            f.write("\n")

            f.write("=== Accuracy by Question Subtype ===\n")
            for q_type, subs in test_subtype_accuracy.items():
                f.write(f"{q_type}:\n")
                for subtype, acc in subs.items():
                    f.write(f"  {subtype}: {acc:.4f}\n")
            f.write("\n")


            f.write("=== Overall Baseline Metrics ===\n")
            for line in baseline[2:]:
                f.write(line)

        print(f"All evaluation results saved to {report_path}.")


if __name__ == "__main__":
    main()