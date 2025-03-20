import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from pathlib import Path
# legend formatting
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

def parse_training_log(log_path):
    """
    Parses the training log and extracts metrics.
    
    args:
        log_path (str): path to the log
    
    returns:
        dict: dictionary containing parsed training data
    """
    with open(log_path, 'r') as f:
        log_text = f.read()
    
    data = {
        'epochs': [],
        'test_results': None
    }
    
    epoch_blocks = re.findall(r'epoch (\d+)/\d+\n(.*?)(?=epoch \d+/\d+|\ntotal training time|\nevaluating on test set)', 
                             log_text, re.DOTALL | re.IGNORECASE)
    
    for epoch_num, epoch_content in epoch_blocks:
        epoch_data = {
            'epoch_number': int(epoch_num),
            'training_loss': None,
            'validation_loss': None,
            'validation_accuracy': None,
            'accuracy_by_type': {},
            'accuracy_by_subtype': {
                'relational': {},
                'non-relational': {}
            }
        }
        
        metrics_match = re.search(r'training loss: ([\d.]+), validation loss: ([\d.]+), validation accuracy: ([\d.]+)', 
                                 epoch_content, re.IGNORECASE)
        if metrics_match:
            epoch_data['training_loss'] = float(metrics_match.group(1))
            epoch_data['validation_loss'] = float(metrics_match.group(2))
            epoch_data['validation_accuracy'] = float(metrics_match.group(3))
        
        type_block_match = re.search(r'accuracy breakdown per question type:\n(.*?)(?=accuracy breakdown per question subtype:)', 
                                    epoch_content, re.DOTALL | re.IGNORECASE)
        if type_block_match:
            type_block = type_block_match.group(1)
            type_matches = re.findall(r'  (\w+-?\w*): ([\d.]+)', type_block)
            for type_name, acc in type_matches:
                epoch_data['accuracy_by_type'][type_name.lower()] = float(acc)
        
        subtype_block_match = re.search(r'accuracy breakdown per question subtype:(.*?)(?=model checkpoint saved|$)', 
                                       epoch_content, re.DOTALL | re.IGNORECASE)
        if subtype_block_match:
            subtype_block = subtype_block_match.group(1)
            
            relational_block_match = re.search(r'  relational:(.*?)(?=  non-relational:|$)', 
                                              subtype_block, re.DOTALL | re.IGNORECASE)
            if relational_block_match:
                relational_block = relational_block_match.group(1)
                rel_matches = re.findall(r'    (\w+): ([\d.]+)', relational_block)
                for subtype, acc in rel_matches:
                    epoch_data['accuracy_by_subtype']['relational'][subtype.lower()] = float(acc)
            
            non_relational_block_match = re.search(r'  non-relational:(.*?)(?=$)', 
                                                  subtype_block, re.DOTALL | re.IGNORECASE)
            if non_relational_block_match:
                non_relational_block = non_relational_block_match.group(1)
                non_rel_matches = re.findall(r'    (\w+): ([\d.]+)', non_relational_block)
                for subtype, acc in non_rel_matches:
                    epoch_data['accuracy_by_subtype']['non-relational'][subtype.lower()] = float(acc)
        
        data['epochs'].append(epoch_data)
    
    test_block_match = re.search(r'evaluating on test set\.\.\.(.*?)(?=final model saved|$)', 
                                log_text, re.DOTALL | re.IGNORECASE)
    if test_block_match:
        test_block = test_block_match.group(1)
        test_data = {
            'loss': None,
            'accuracy': None,
            'accuracy_by_type': {},
            'accuracy_by_subtype': {
                'relational': {},
                'non-relational': {}
            }
        }
        
        test_metrics_match = re.search(r'test loss: ([\d.]+), test accuracy: ([\d.]+)', test_block, re.IGNORECASE)
        if test_metrics_match:
            test_data['loss'] = float(test_metrics_match.group(1))
            test_data['accuracy'] = float(test_metrics_match.group(2))
        
        test_type_block_match = re.search(r'test accuracy breakdown per question type:\n(.*?)(?=test accuracy breakdown per question subtype:)', 
                                         test_block, re.DOTALL | re.IGNORECASE)
        if test_type_block_match:
            test_type_block = test_type_block_match.group(1)
            test_type_matches = re.findall(r'  (\w+-?\w*): ([\d.]+)', test_type_block)
            for type_name, acc in test_type_matches:
                test_data['accuracy_by_type'][type_name.lower()] = float(acc)
        
        test_subtype_block_match = re.search(r'test accuracy breakdown per question subtype:(.*?)(?=final model saved|$)', 
                                            test_block, re.DOTALL | re.IGNORECASE)
        if test_subtype_block_match:
            test_subtype_block = test_subtype_block_match.group(1)
            
            test_rel_block_match = re.search(r'  relational:(.*?)(?=  non-relational:|$)', 
                                            test_subtype_block, re.DOTALL | re.IGNORECASE)
            if test_rel_block_match:
                test_rel_block = test_rel_block_match.group(1)
                test_rel_matches = re.findall(r'    (\w+): ([\d.]+)', test_rel_block)
                for subtype, acc in test_rel_matches:
                    test_data['accuracy_by_subtype']['relational'][subtype.lower()] = float(acc)
            
            test_non_rel_block_match = re.search(r'  non-relational:(.*?)(?=$)', 
                                                test_subtype_block, re.DOTALL | re.IGNORECASE)
            if test_non_rel_block_match:
                test_non_rel_block = test_non_rel_block_match.group(1)
                test_non_rel_matches = re.findall(r'    (\w+): ([\d.]+)', test_non_rel_block)
                for subtype, acc in test_non_rel_matches:
                    test_data['accuracy_by_subtype']['non-relational'][subtype.lower()] = float(acc)
        
        data['test_results'] = test_data
    
    return data

def create_dataframes(parsed_data):
    """
    Converts parsed log data into pandas dataframes for analysis and plotting.
    
    args:
        parsed_data (dict): parsed log data
    
    returns:
        tuple: tuple of dataframes (epochs_df, test_df)
    """
    epochs_data = []
    for epoch in parsed_data['epochs']:
        epoch_row = {
            'epoch': epoch['epoch_number'],
            'training_loss': epoch['training_loss'],
            'validation_loss': epoch['validation_loss'],
            'validation_accuracy': epoch['validation_accuracy']
        }
        
        for type_name, acc in epoch['accuracy_by_type'].items():
            epoch_row[f'acc_{type_name}'] = acc
        
        for type_name, subtypes in epoch['accuracy_by_subtype'].items():
            for subtype, acc in subtypes.items():
                epoch_row[f'acc_{type_name}_{subtype}'] = acc
        
        epochs_data.append(epoch_row)
    
    epochs_df = pd.DataFrame(epochs_data)
    
    test_data = {}
    if parsed_data['test_results']:
        test_data = {
            'loss': parsed_data['test_results']['loss'],
            'accuracy': parsed_data['test_results']['accuracy']
        }
        
        for type_name, acc in parsed_data['test_results']['accuracy_by_type'].items():
            test_data[f'acc_{type_name}'] = acc
        
        for type_name, subtypes in parsed_data['test_results']['accuracy_by_subtype'].items():
            for subtype, acc in subtypes.items():
                test_data[f'acc_{type_name}_{subtype}'] = acc
    
    test_df = pd.DataFrame([test_data]) if test_data else None
    
    return epochs_df, test_df

def visualize_training_log(log_path, output_dir=None):
    """
    Parses a training log and creates visualizations.
    
    args:
        log_path (str): path to the log file
        output_dir (str, optional): directory to save the output plots. if none, plots will be displayed.
    
    returns:
        tuple: tuple of parsed data and dataframes (parsed_data, epochs_df, test_df)
    """
    parsed_data = parse_training_log(log_path)
    epochs_df, test_df = create_dataframes(parsed_data)
    
    # define column lists for plots
    rel_subtype_cols = [col for col in epochs_df.columns if col.startswith('acc_relational_')]
    non_rel_subtype_cols = [col for col in epochs_df.columns if col.startswith('acc_non-relational_')]
    type_cols = [col for col in epochs_df.columns if col.startswith('acc_') and '_' not in col.replace('acc_', '', 1)]
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_theme(font_scale=1.2)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # plot 1: loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df['epoch'], epochs_df['training_loss'], 'o-', label='training loss', linewidth=2)
    plt.plot(epochs_df['epoch'], epochs_df['validation_loss'], 's-', label='validation loss', linewidth=2)
    if test_df is not None and 'loss' in test_df.columns:
        # square marker, vertical alignment
        last_x = epochs_df['epoch'].iloc[-1]
        plt.plot(last_x + 0.25, test_df['loss'].iloc[0], 's', 
                markersize=10, markeredgewidth=2, color='red',
                label=f'test loss: {test_df["loss"].iloc[0]:.4f}')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs_df['epoch'])
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(output_path / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # plot 2: accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_df['epoch'], epochs_df['validation_accuracy'], 'o-', label='validation accuracy', linewidth=2)
    if test_df is not None and 'accuracy' in test_df.columns:
        last_x = epochs_df['epoch'].iloc[-1]
        plt.plot(last_x + 0.25, test_df['accuracy'].iloc[0], 's', 
                markersize=10, markeredgewidth=2, color='red',
                label=f'test accuracy: {test_df["accuracy"].iloc[0]:.4f}')
    plt.title('Validation accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs_df['epoch'])
    plt.ylim(0, max(epochs_df['validation_accuracy'].max() * 1.1, 
                   test_df['accuracy'].iloc[0] * 1.1 if test_df is not None and 'accuracy' in test_df.columns else 0))
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(output_path / 'accuracy_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # plot 3: accuracy by question type
    type_cols = [col for col in epochs_df.columns if col.startswith('acc_') and '_' not in col.replace('acc_', '', 1)]
    
    if type_cols:
        plt.figure(figsize=(10, 6))
        
        # custom legend
        legend_elements = []
        
        for col in type_cols:
            type_name = col.replace('acc_', '')
            line = plt.plot(epochs_df['epoch'], epochs_df[col], 'o-', linewidth=2)[0]
            line_color = line.get_color()

            test_value = f": {test_df[col].iloc[0]:.4f}" if test_df is not None and col in test_df.columns else ""
            legend_label = f"{type_name.lower()} (test{test_value})"
            
            legend_elements.append((line, Line2D([0], [0], marker='s', color='w', markerfacecolor=line_color, 
                                              markeredgecolor=line_color, markersize=10, markeredgewidth=2), 
                                  legend_label))
            
            if test_df is not None and col in test_df.columns:
                plt.plot(epochs_df['epoch'].iloc[-1] + 0.25, test_df[col].iloc[0], 
                        's', markersize=10, markeredgewidth=2, color=line_color)
                        
        plt.title('Validation accuracy by question type')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(epochs_df['epoch'])
        plt.ylim(0, max(epochs_df[type_cols].max().max() * 1.1,
                       test_df[type_cols].max().max() * 1.1 if test_df is not None else 0))
        
        plt.legend(handles=[tuple(elements[:2]) for elements in legend_elements],
                 labels=[elements[2] for elements in legend_elements],
                 handler_map={tuple: HandlerTuple(ndivide=None)})
        
        plt.grid(True)
        if output_dir:
            plt.savefig(output_path / 'accuracy_by_type.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    if rel_subtype_cols:
        plt.figure(figsize=(10, 6))

        legend_elements = []
        
        for col in rel_subtype_cols:
            subtype = col.replace('acc_relational_', '')

            line = plt.plot(epochs_df['epoch'], epochs_df[col], 'o-', linewidth=2)[0]
            line_color = line.get_color()
            
            test_value = f": {test_df[col].iloc[0]:.4f}" if test_df is not None and col in test_df.columns else ""
            legend_label = f"{subtype.lower()} (test{test_value})"
            
            legend_elements.append((line, Line2D([0], [0], marker='s', color='w', markerfacecolor=line_color, 
                                              markeredgecolor=line_color, markersize=10, markeredgewidth=2), 
                                  legend_label))
            
            if test_df is not None and col in test_df.columns:
                plt.plot(epochs_df['epoch'].iloc[-1] + 0.25, test_df[col].iloc[0], 
                        's', markersize=10, markeredgewidth=2, color=line_color)
                        
        plt.title('Validation accuracy by relational question subtype')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(epochs_df['epoch'])
        plt.ylim(0, max(epochs_df[rel_subtype_cols].max().max() * 1.1,
                       test_df[rel_subtype_cols].max().max() * 1.1 if test_df is not None else 0))
        
        plt.legend(handles=[tuple(elements[:2]) for elements in legend_elements],
                 labels=[elements[2] for elements in legend_elements],
                 handler_map={tuple: HandlerTuple(ndivide=None)})
        
        plt.grid(True)
        if output_dir:
            plt.savefig(output_path / 'accuracy_relational_subtypes.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    if non_rel_subtype_cols:
        plt.figure(figsize=(10, 6))

        legend_elements = []
        
        for col in non_rel_subtype_cols:
            subtype = col.replace('acc_non-relational_', '')
            line = plt.plot(epochs_df['epoch'], epochs_df[col], 'o-', linewidth=2)[0]
            line_color = line.get_color()
            
            test_value = f": {test_df[col].iloc[0]:.4f}" if test_df is not None and col in test_df.columns else ""
            legend_label = f"{subtype.lower()} (test{test_value})"
            
            legend_elements.append((line, Line2D([0], [0], marker='s', color='w', markerfacecolor=line_color, 
                                              markeredgecolor=line_color, markersize=10, markeredgewidth=2), 
                                  legend_label))
            
            if test_df is not None and col in test_df.columns:
                plt.plot(epochs_df['epoch'].iloc[-1] + 0.25, test_df[col].iloc[0], 
                        's', markersize=10, markeredgewidth=2, color=line_color)
                        
        plt.title('Validation accuracy by non-relational question subtype')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(epochs_df['epoch'])
        plt.ylim(0, max(epochs_df[non_rel_subtype_cols].max().max() * 1.1,
                       test_df[non_rel_subtype_cols].max().max() * 1.1 if test_df is not None else 0))
        
        plt.legend(handles=[tuple(elements[:2]) for elements in legend_elements],
                 labels=[elements[2] for elements in legend_elements],
                 handler_map={tuple: HandlerTuple(ndivide=None)})
        
        plt.grid(True)
        if output_dir:
            plt.savefig(output_path / 'accuracy_non_relational_subtypes.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    # dashboard
    plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 2, figure=plt.gcf())
    
    # plot 1: training and validation losses (top left)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(epochs_df['epoch'], epochs_df['training_loss'], 'o-', label='training loss', linewidth=2)
    ax1.plot(epochs_df['epoch'], epochs_df['validation_loss'], 's-', label='validation loss', linewidth=2)
    
    if test_df is not None and 'loss' in test_df.columns:
        test_loss = test_df['loss'].iloc[0]
        last_x = epochs_df['epoch'].iloc[-1]
        ax1.plot(last_x + 0.25, test_loss, 's', markersize=10, markeredgewidth=2, 
                color='red', label=f'test loss: {test_loss:.4f}')
    
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(epochs_df['epoch'])
    ax1.legend()
    ax1.grid(True)
    
    # plot 2: overall validation accuracy (top right)
    ax2 = plt.subplot(gs[0, 1])
    legend_elements_acc = []
    
    validation_acc_line = ax2.plot(epochs_df['epoch'], epochs_df['validation_accuracy'], 'o-', linewidth=2)[0]
    legend_elements_acc.append(validation_acc_line)
    
    if test_df is not None and 'accuracy' in test_df.columns:
        test_acc = test_df['accuracy'].iloc[0]
        last_x = epochs_df['epoch'].iloc[-1]
        test_acc_marker = ax2.plot(last_x + 0.25, test_acc, 's', markersize=10, markeredgewidth=2,
                color='red')[0]
        legend_elements_acc.append(test_acc_marker)
    
    ax2.set_title('Overall validation accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(epochs_df['epoch'])
    ax2.set_ylim(0, max(epochs_df['validation_accuracy'].max() * 1.1, 
                       test_df['accuracy'].iloc[0] * 1.1 if test_df is not None and 'accuracy' in test_df.columns else 0))
    
    legend_labels_acc = ['validation accuracy']
    if test_df is not None and 'accuracy' in test_df.columns:
        legend_labels_acc.append(f'test accuracy: {test_df["accuracy"].iloc[0]:.4f}')
        
    ax2.legend(legend_elements_acc, legend_labels_acc)
    ax2.grid(True)
    
    # plot 3: relational vs non-relational validation accuracy (middle row, both cols)
    ax3 = plt.subplot(gs[1, 0:2])
    
    # get relational and non-relational columns
    rel_col = [col for col in epochs_df.columns if col == 'acc_relational']
    non_rel_col = [col for col in epochs_df.columns if col == 'acc_non-relational']
    
    legend_elements_rel_vs_nonrel = []
    
    if rel_col:
        line_rel = ax3.plot(epochs_df['epoch'], epochs_df[rel_col[0]], 'o-', label='relational', linewidth=2)[0]
        line_color_rel = line_rel.get_color()
        
        if test_df is not None and rel_col[0] in test_df.columns:
            test_rel_acc = test_df[rel_col[0]].iloc[0]
            last_x = epochs_df['epoch'].iloc[-1]
            ax3.plot(last_x + 0.25, test_rel_acc, 's', markersize=10, markeredgewidth=2, 
                   color=line_color_rel)
            
            legend_elements_rel_vs_nonrel.append((line_rel, 
                                             Line2D([0], [0], marker='s', color='w', 
                                                   markerfacecolor=line_color_rel, 
                                                   markeredgecolor=line_color_rel, 
                                                   markersize=10, markeredgewidth=2),
                                             f"relational (test: {test_rel_acc:.4f})"))
        else:
            legend_elements_rel_vs_nonrel.append((line_rel, None, "relational"))
    
    if non_rel_col:
        line_nonrel = ax3.plot(epochs_df['epoch'], epochs_df[non_rel_col[0]], 's-', label='non-relational', linewidth=2)[0]
        line_color_nonrel = line_nonrel.get_color()
        
        if test_df is not None and non_rel_col[0] in test_df.columns:
            test_non_rel_acc = test_df[non_rel_col[0]].iloc[0]
            last_x = epochs_df['epoch'].iloc[-1]
            ax3.plot(last_x + 0.25, test_non_rel_acc, 's', markersize=10, markeredgewidth=2, 
                   color=line_color_nonrel)
            
            legend_elements_rel_vs_nonrel.append((line_nonrel, 
                                             Line2D([0], [0], marker='s', color='w', 
                                                   markerfacecolor=line_color_nonrel, 
                                                   markeredgecolor=line_color_nonrel, 
                                                   markersize=10, markeredgewidth=2),
                                             f"non-relational (test: {test_non_rel_acc:.4f})"))
        else:
            legend_elements_rel_vs_nonrel.append((line_nonrel, None, "non-relational"))
    
    ax3.set_title('Relational vs non-relational validation accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticks(epochs_df['epoch'])
    ax3.set_xticks(epochs_df['epoch'])
    
    if rel_col or non_rel_col:
        cols_to_check = rel_col + non_rel_col
        max_val = epochs_df[cols_to_check].max().max()
        
        test_vals = []
        if test_df is not None:
            for col in cols_to_check:
                if col in test_df.columns:
                    test_vals.append(test_df[col].iloc[0])
        
        if test_vals:
            max_val = max(max_val, max(test_vals))
            
        ax3.set_ylim(0, max_val * 1.1)
    
    formatted_handles = []
    formatted_labels = []
    
    for elements in legend_elements_rel_vs_nonrel:
        if len(elements) == 3 and elements[1] is not None:
            # if we have both line and marker
            formatted_handles.append((elements[0], elements[1]))
            formatted_labels.append(elements[2])
        else:
            # just the line
            formatted_handles.append(elements[0])
            formatted_labels.append(elements[2])
    
    if any(isinstance(h, tuple) for h in formatted_handles):
        # we have at least one combined handle
        ax3.legend(handles=formatted_handles, labels=formatted_labels, 
                 handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        # just regular handles
        ax3.legend(formatted_handles, formatted_labels)
    
    ax3.grid(True)
    
    # plot 4: relational question subtypes (bottom left)
    ax4 = plt.subplot(gs[2, 0])
    rel_subtype_cols = [col for col in epochs_df.columns if col.startswith('acc_relational_')]
    
    legend_elements = []
    
    for col in rel_subtype_cols:
        subtype = col.replace('acc_relational_', '')
        line = ax4.plot(epochs_df['epoch'], epochs_df[col], 'o-', linewidth=2)[0]
        line_color = line.get_color()
        
        test_value = f": {test_df[col].iloc[0]:.4f}" if test_df is not None and col in test_df.columns else ""
        legend_label = f"{subtype} (test{test_value})"
        
        legend_elements.append((line, Line2D([0], [0], marker='s', color='w', markerfacecolor=line_color, 
                                         markeredgecolor=line_color, markersize=10, markeredgewidth=2), 
                             legend_label))
        
        if test_df is not None and col in test_df.columns:
            test_subtype_acc = test_df[col].iloc[0]
            last_x = epochs_df['epoch'].iloc[-1]
            ax4.plot(last_x + 0.25, test_subtype_acc, 's', markersize=10, markeredgewidth=2, 
                    color=line_color)
    
    ax4.set_title('Validation accuracy - relational question subtypes')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_xticks(epochs_df['epoch'])
    
    # y-axis limit based on available data
    if rel_subtype_cols:
        max_val = epochs_df[rel_subtype_cols].max().max()
        
        # check test values too
        test_vals = []
        if test_df is not None:
            for col in rel_subtype_cols:
                if col in test_df.columns:
                    test_vals.append(test_df[col].iloc[0])
        
        if test_vals:
            max_val = max(max_val, max(test_vals))
            
        ax4.set_ylim(0, max_val * 1.1)
    
    ax4.legend(handles=[tuple(elements[:2]) for elements in legend_elements],
             labels=[elements[2] for elements in legend_elements],
             handler_map={tuple: HandlerTuple(ndivide=None)})
    
    ax4.grid(True)
    
    # plot 5: non-relational question subtypes (bottom right)
    ax5 = plt.subplot(gs[2, 1])
    non_rel_subtype_cols = [col for col in epochs_df.columns if col.startswith('acc_non-relational_')]
    
    legend_elements = []
    
    for col in non_rel_subtype_cols:
        subtype = col.replace('acc_non-relational_', '')
        line = ax5.plot(epochs_df['epoch'], epochs_df[col], 's-', linewidth=2)[0]
        line_color = line.get_color()
        
        test_value = f": {test_df[col].iloc[0]:.4f}" if test_df is not None and col in test_df.columns else ""
        legend_label = f"{subtype} (test{test_value})"
        
        legend_elements.append((line, Line2D([0], [0], marker='s', color='w', markerfacecolor=line_color, 
                                         markeredgecolor=line_color, markersize=10, markeredgewidth=2), 
                             legend_label))
        
        if test_df is not None and col in test_df.columns:
            test_subtype_acc = test_df[col].iloc[0]
            last_x = epochs_df['epoch'].iloc[-1]
            ax5.plot(last_x + 0.25, test_subtype_acc, 's', markersize=10, markeredgewidth=2, 
                    color=line_color)
    
    ax5.set_title('Validation accuracy - relational question subtypes')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Accuracy')
    ax5.set_xticks(epochs_df['epoch'])
    
    # set y-axis limit based on available data
    if non_rel_subtype_cols:
        max_val = epochs_df[non_rel_subtype_cols].max().max()
        
        # check test values too
        test_vals = []
        if test_df is not None:
            for col in non_rel_subtype_cols:
                if col in test_df.columns:
                    test_vals.append(test_df[col].iloc[0])
        
        if test_vals:
            max_val = max(max_val, max(test_vals))
            
        ax5.set_ylim(0, max_val * 1.1)
    
    ax5.legend(handles=[tuple(elements[:2]) for elements in legend_elements],
             labels=[elements[2] for elements in legend_elements],
             handler_map={tuple: HandlerTuple(ndivide=None)})
    
    ax5.grid(True)
    
    plt.tight_layout()
    
    plt.suptitle(f'Training overview - {Path(log_path)}', fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.94, bottom=0.12)  # adjust bottom to make room for the legend
    
    if output_dir:
        plt.savefig(output_path / 'training_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return parsed_data, epochs_df, test_df