import argparse
import json
import os
from io import StringIO

import numpy as np
import pandas as pd

columns = [
    "Team",
    "Goals",
    "Shots",
    "Fouls",
    "Yellow Cards",
    "Red Cards",
    "Corner Kicks",
    "Free Kicks",
    "Offsides",
]

difficulty = {
    "Goals": 0,
    "Red Cards": 0,
    "Yellow Cards": 1,
    "Corner Kicks": 1,
    "Free Kicks": 1,
    "Offsides": 1,
    "Shots": 2,
    "Fouls": 2,
}

cnt0 = [0, 0, 0, 0]
for i in difficulty.keys():
    cnt0[difficulty[i]] += 1


def evaluate(data_dir, output_dir):
    with open(os.path.join(data_dir, "test.json"), "r") as f:
        test_file = json.load(f)
    all_ground_truth = {}
    for inst in test_file:
        all_ground_truth[inst["id"]] = pd.read_csv(
            StringIO(inst["table"].replace("<NEWLINE>", "\n"))
        )
    result = []
    qq = os.listdir(output_dir)
    qq.sort()
    for file_name in qq:
        if ".csv" in file_name:
            idx = int(file_name.split(".")[0])
            try:
                output = pd.read_csv(os.path.join(output_dir, file_name))
                ground_truth = all_ground_truth[idx]
                res = []
                correct_col = 0
                mses = [0, 0, 0, 0]
                maes = [0, 0, 0, 0]
                accs = [0, 0, 0, 0]
                for column in columns[1:]:
                    output_col = list(output[column])
                    output_col = np.array(output_col)
                    output_col = [o if type(o) != np.str_ else 0 for o in output_col]
                    output_col = np.array(output_col)
                    output_col = np.nan_to_num(output_col)
                    ground_truth_col = list(ground_truth[column])
                    ground_truth_col = np.array(ground_truth_col)
                    ground_truth_col = [
                        o if type(o) != np.str_ else 0 for o in ground_truth_col
                    ]
                    ground_truth_col = np.array(ground_truth_col)
                    ground_truth_col = np.nan_to_num(ground_truth_col)
                    eid = difficulty[column]
                    mses[eid] += np.square(output_col - ground_truth_col).sum()
                    maes[eid] += np.abs(output_col - ground_truth_col).sum()
                    accs[eid] += (output_col == ground_truth_col).sum()
                    mses[3] += np.square(output_col - ground_truth_col).sum()
                    maes[3] += np.abs(output_col - ground_truth_col).sum()
                    accs[3] += (output_col == ground_truth_col).sum()
                res.extend(
                    [
                        (mses[0] / (cnt0[0] * 2)) ** 0.5,
                        (maes[0] / (cnt0[0] * 2)),
                        100 - (accs[0] / (cnt0[0] * 2)) * 100,
                        (mses[1] / (cnt0[1] * 2)) ** 0.5,
                        (maes[1] / (cnt0[1] * 2)),
                        100 - (accs[1] / (cnt0[1] * 2)) * 100,
                        (mses[2] / (cnt0[2] * 2)) ** 0.5,
                        (maes[2] / (cnt0[2] * 2)),
                        100 - (accs[2] / (cnt0[2] * 2)) * 100,
                        (mses[3] / 16) ** 0.5,
                        (maes[3] / 16),
                        100 - (accs[3] / 16) * 100,
                    ]
                )
            except Exception as e:
                print(line)
                raise ValueError(e)
            result.append(res)
    print("test {} tables for {}".format(len(result), output_dir))
    return pd.DataFrame(
        result,
        columns=[
            "Easy-RMSE",
            "Easy-MAE",
            "Easy-EM",
            "Medium-RMSE",
            "Medium-MAE",
            "Medium-EM",
            "Hard-RMSE",
            "Hard-MAE",
            "Hard-EM",
            "AVG-RMSE",
            "AVG-MAE",
            "AVG-EM",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", type=str, required=True, help="Path to the data folder"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output folder"
    )
    args = parser.parse_args()

    result = evaluate(args.data, args.output)
    print(result.describe().loc["mean"])
    pass
