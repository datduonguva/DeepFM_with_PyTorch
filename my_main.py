"""
Load data and train model
"""
import os
import json
from collections import namedtuple, Counter

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = "data"
def read_training_data():
    """
    Read the first 1M rows of data to find the categories 
    """
    category = {
        "label": []
    }
    category.update(
        {
            f"i{i}": [] for i in range(13)
        }
    )
    category.update(
        {
            f"c{i}": set([]) for i in range(26)
        }
    )
 
    with open(os.path.join(ROOT, "train.txt"), "r") as f:
        for i, line in tqdm(enumerate(f)):

            line = line.split("\t")

            for j in range(13):
                if line[j + 1].isnumeric():
                    val = int(line[j + 1])
                    category[f"i{j}"].append(val)

            for j in range(26):
                category[f"c{j}"].add(line[j + 14].strip())

            if i == 1000000:
                break

    for j in range(26):
        category[f"c{j}"] = sorted(category[f"c{j}"])

    for j in range(13):
        category[f"i{j}"] = {
            "min": int(np.min(category[f"i{j}"])),
            "max": int(np.max(category[f"i{j}"])),
            "mean": int(np.mean(category[f"i{j}"])),
            "median": int(np.median(category[f"i{j}"])),
            "std": int(np.std(category[f"i{j}"])),
        }


    with open("data/stats_1M.json", "w") as f:
        json.dump(category, f, indent=2)


class DataGenenerator():
    def __init__(self, stat_path: str):
        """
        load the set of classes for categorical fields, the std for continuous fields
        """ 
        with open(stat_path, "r") as f:
            categories = json.load(f)
            for j in range(26):
                keys = categories[f"c{j}"][:3]
                categories[f"c{j}"] = {_: i for i, _ in enumerate(sorted(categories[f"c{j}"]))}


        # for numeric: turn to Z score: cap at [-2, 2] 
        # for categorical: 1-hot encode

        encoded_features = []
        outputs = []
        data = pd.read_csv(
            os.path.join(ROOT, "train.txt"), delimiter="\t", header=None, nrows=10000
        )

        output = data[0]

        def scale(value: float, stats: dict) -> float:
            mean = stats['mean']
            std = stats['std']
            median = stats['median']

            if np.isnan(value):
                return median
            elif std == 0:
                return value
            return np.clip((value - mean)/std, -3, 3)

        for j in range(13):
            data[1 + j] = data[1 + j].apply(
                lambda val: scale(val, categories[f"i{j}"])
            )
        self.continuous_values = data[1: 1 + 13].values


        for j in range(26):
            data[14 + j]  = data[14 + j].apply(lambda val: categories[f"c{j}"].get(val, -1))

        self.descrete_values = data[14 + j : ].values

        self.feature_sizes = [1] * 13 + [
            len(categories[f"c{j}"]) for j in range(26)
        ]

        print(self.feature_sizes)



class DeepFM(torch.nn.Module):
    
    def __init__(self, feature_sizes, embedding_size, device=None):
        super().__init__()
        """
        If feature has size 1, it is numeric, else, it is categorical
        """
        self.feature_sizes = feature_sizes
        self.fm1_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(feature_size, 1) for feature_size in feature_sizes ]
        )
        self.fm2_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(feature_size,  embedding_size, device=device)
                for feature_size in feature_sizes
            ]
        )
        
        self.deep_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=in_, out_features=out_)
            for in_, out_ in [
                (len(feature_sizes) * embedding_size, 1024),
                (1024, 128),
                (128, 32),
                (32, 1)
            ]
        ])


        

    def forward(self, x_i, x_v):
        """
        x_i: (N, m): the positions  of each features
        x_v: (N, m): the values of each features
        """
        batch = x_i.shape[0]

        # emb(x_i[:, i]) = (N, 1) scaled by x_v[:, i] (N, 1) -> (N, 1) -> (N, 1)
        # after concatenate, it has (N, m)
        fm1_output = torch.cat(
            [
                emb(x_i[:, i])*x_v[:, i].reshape((-1, 1))  #  (N, 1)
                for i, emb in enumerate(self.fm1_embeddings)
            ],
            axis=1
        ) 
        print("fm1_output: ", fm1_output.shape)

        # each is a (N, K, m) vector
        scaled_embeddings = torch.concat([
                (emb(x_i[:, i])*x_v[:, i].reshape((-1, 1))).unsqueeze(-1)
                for i, emb in enumerate(self.fm2_embeddings)
            ], 
            axis=2
        )
        print("scaled_embeddings: ", scaled_embeddings.shape)
        
        # square_of_sum (N, K)
        square_of_sum = torch.sum(scaled_embeddings, axis=2)
        square_of_sum = square_of_sum * square_of_sum  # (N, K)

        # sum of square
        sum_of_square = torch.sum(scaled_embeddings * scaled_embeddings, axis=2)

        # interaction part (N, K)
        interactions = 0.5*(square_of_sum - sum_of_square) 

        # deep part:
        x = torch.reshape(scaled_embeddings, (batch, -1))
        for deep_layer in self.deep_layers:
            x = deep_layer(x)
            x = torch.sigmoid(x)

        # (N, 1)
        output = torch.sigmoid(
            torch.sum(fm1_output, dim=1, keepdim=True) + 
            torch.sum(interactions, dim=1, keepdim=True) + x
        )

        return output
if __name__ == '__main__':
    
    if False:
        read_training_data()

    if False:

        data_generator = DataGenenerator(
            stat_path="data/stats_1M.json"
        )

    if True:
        model = DeepFM([1, 1, 4, 5, 6], 4)

        x_i = torch.Tensor([
            [0, 0, 1, 2, 3],
            [0, 0, 2, 3, 4],
            [0, 0, 3, 4 ,5]
        ]).to(torch.long)

        x_v = torch.Tensor([
            [1, 2, 1, 1, 1],
            [-1, -2, 1, 1, 1],
            [4, 5, 1, 1, 1],
        ])

        output = model.forward(x_i, x_v)
        print(output)
