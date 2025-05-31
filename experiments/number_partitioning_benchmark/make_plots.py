from __future__ import annotations

import functools
import json
from collections import defaultdict
from typing import TYPE_CHECKING
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    data_type = dict[str, tuple[NDArray[np.int_], NDArray[np.float64]]]

data_folder = Path(__file__).parent / "data"



def read_and_process_data() -> dict[int, data_type]:
    all_data = defaultdict(dict) 

    for filepath in data_folder.iterdir():
        filename = filepath.name.removesuffix(".json")
        number, cd_type, noise = filename.split("_")
        n_numbers = int(number.removesuffix("numbers"))
        noise = float(noise.removeprefix("noise"))

        json_text = filepath.read_text(encoding="utf-8")
        data = json.loads(json_text)
        all_data[n_numbers][f"{cd_type} {noise}"] = (
            np.array(data.pop("n_layers"), dtype=int),
            np.mean(np.vstack(list(data.values())), axis=0)
        )
    return all_data

def line_plot(n_numbers: int, data: data_type) -> None:
    fig, ax = plt.subplots()
    for label, (n_layers, mean_succ) in data.items(): 
        ax.plot(n_layers, mean_succ, label=label)

    ax.hlines(2 ** (1-n_numbers), n_layers[0], n_layers[-1], label="Random")
    ax.legend()
    fig.savefig(Path(__file__).parent / "figures" / f"{n_numbers}numbers.jpg")

def bar_plot(n_numbers: int, data: data_type) -> None:
    fig, ax = plt.subplots()

    all_noise = sorted({float(label.rsplit(maxsplit=1)[1]) for label in data})
    all_cd_types = sorted({label.rsplit(maxsplit=1)[0] for label in data})

    default_value = functools.partial(np.zeros_like, all_noise)
    y_data = defaultdict(default_value)

    for label, (_, mean_succ) in data.items():
        best_succ = np.max(mean_succ)
        cd_type, noise_str = label.rsplit(maxsplit=1)
        noise = float(noise_str)
        y_data[cd_type][all_noise.index(noise)] = best_succ

    bar_width = 1 / (len(all_cd_types)+1)
    x_data = {cd_type : np.arange(len(all_noise)) + i*bar_width for i, cd_type in enumerate(all_cd_types)}

    for cd_type in all_cd_types:
        ax.bar(x_data[cd_type], y_data[cd_type], width=bar_width, label=cd_type)

    mean_x_data = np.mean(np.vstack(list(x_data.values())), axis=0)
    ax.hlines(2**(1-n_numbers), mean_x_data[0]-1, mean_x_data[-1]+1, label="Random") 
    
    ax.set_xticks(mean_x_data, all_noise)
    ax.set_xlim(mean_x_data[0]-1, mean_x_data[-1]+1)
    ax.set_xlabel("noise")
    ax.set_ylabel("succes probability")
    ax.legend()

    fig.savefig(Path(__file__).parent / "figures" / f"{n_numbers}numbers_bar.jpg")
    

    print(y_data)

all_data = read_and_process_data()
for n_numbers, data in all_data.items():
    line_plot(n_numbers, data)
    bar_plot(n_numbers, data)
