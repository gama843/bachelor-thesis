import os
import json
import argparse
from utils import plot_accuracy_breakdown
from collections import defaultdict

def process_results_file(results_path):
    with open(results_path, "r") as f:
        data = json.load(f)
    return data["accuracySummary"]

def main(human_root_dir):
    human_dirs = [d for d in os.listdir(human_root_dir)
                  if os.path.isdir(os.path.join(human_root_dir, d))]
    
    aggregated = defaultdict(list)

    for human_dir in human_dirs:
        folder_path = os.path.join(human_root_dir, human_dir)
        results_path = os.path.join(folder_path, "results.json")
        if not os.path.exists(results_path):
            print(f"Skipping {human_dir}: results.json not found.")
            continue
        
        acc = process_results_file(results_path)
        overall = acc.get("overall", 0.0)

        subtypes = {k: v for k, v in acc.items() if k in {"shape", "topbottom", "leftright", "closest", "farthest", "count"}}
        types = {k: v for k, v in acc.items() if k in {"relational", "non-relational"}}

        aggregated["overall"].append(overall)
        for k, v in types.items():
            aggregated[k].append(v)
        for k, v in subtypes.items():
            aggregated[k].append(v)

        plot_accuracy_breakdown(
            overall_accuracy=overall,
            type_acc=types,
            subtype_acc=subtypes,
            model_name=f"human_{human_dir}",
            output_dir=folder_path
        )
        print(f"Plotted individual results for {human_dir}")

    if aggregated:
        avg_acc = {k: sum(vs)/len(vs) for k, vs in aggregated.items()}
        overall = avg_acc.pop("overall", 0.0)

        type_acc = {k: v for k, v in avg_acc.items() if k in {"relational", "non-relational"}}
        subtype_acc = {k: v for k, v in avg_acc.items() if k not in type_acc}

        plot_path = plot_accuracy_breakdown(
            overall_accuracy=overall,
            type_acc=type_acc,
            subtype_acc=subtype_acc,
            model_name="human_average",
            output_dir=human_root_dir
        )
        print(f"Plotted average human performance at: {plot_path}")

        output_json = {
            "source": "aggregated_humans",
            "accuracySummary": {
                **subtype_acc,
                **type_acc,
                "overall": overall
            }
        }
        json_path = os.path.join(human_root_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"Averaged results written to: {json_path}")
    else:
        print("No valid results.json files found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance plots for human evaluation data.")
    parser.add_argument("human_root_dir", type=str, help="Path to the root folder containing human result subfolders.")
    args = parser.parse_args()
    main(args.human_root_dir)
