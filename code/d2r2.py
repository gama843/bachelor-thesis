import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_generator import DataGenerator
from models import ModelConstructor
from dataset_builder import DatasetBuilder, Rotate180DegreesTransform
from training import train_one_epoch, validate_one_epoch
import os

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

    args = parser.parse_args()

    if not any(vars(args).values()):
        # if no arguments are passed, run the default behavior (dataset generation and model training)
        print("No arguments provided. Running default dataset generation and model training.")
        
        generator = DataGenerator('../data')
        generator.generate_dataset(img_dim=128, num_images=100)

        builder = DatasetBuilder('../data', Rotate180DegreesTransform(), 0.5)
        train_loader = DataLoader(builder.train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(builder.val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(builder.test_dataset, batch_size=1, shuffle=False)

        model_constructor = ModelConstructor()
        model = model_constructor.load_model(
            model_type='baseline',
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

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 1

        # training loop
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f'Training Loss: {train_loss:.4f}')

            val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # test after training
        test_loss, test_accuracy = validate_one_epoch(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    else:
        # handle the provided arguments logic
        if args.generate2D:
            generator = DataGenerator(args.generate2D)
            generator.generate_dataset(img_dim=128, num_images=100)
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