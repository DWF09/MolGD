import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def line(df):
    # x = range(1,len(s)+1)
    fig, ax1 = plt.subplots()
    ax1.plot(df["step"], df["3D_FCD"], color='magenta', linestyle="-", linewidth=0.5, label="3D_FCD", marker="o", markersize=2)
    ax1.plot(df["step"], df["2D_FCD"], color='g', linestyle="-", linewidth=0.5, label="2D_FCD", marker="o", markersize=2)
    ax1.set_xlabel("step")
    ax1.set_ylabel('FCD') 
    ax1.set_ylim(0, 20)
    ax1.set_yticks(np.arange(0, 20.01, 2))
    
    ax2 = ax1.twinx()  
    ax2.plot(df["step"], df["3D_mol_stability"], color='r', linestyle="-", linewidth=0.5, label="3D_mol_stability", marker="o", markersize=2)
    ax2.set_ylabel('3D_mol_stability') 
    ax2.set_ylim(0, 1.01)
    ax2.set_yticks(np.arange(0, 1.01, 0.1))
    
    plt.title('FCD and stability')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, labelcolor='linecolor')
    plt.savefig("./QM9.png")

df = pd.read_csv('./QM9.csv')
line(df)

