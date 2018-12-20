import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm


def main():
    df = pd.read_csv('results.csv')
    ns: np.array = np.sort(df['n'].unique())

    cmap = matplotlib.cm.get_cmap('Blues')
    colormap = [cmap(ix) for ix in np.linspace(0, 1, ns.size).tolist()]
    patches_gpu = [mpatches.Patch(color=color, label=n, linestyle='--', hatch='o') for (color, n) in
                   zip(colormap, [f"GPU n={i}" for i in ns])]

    cmap = matplotlib.cm.get_cmap('Reds')
    colormap = [cmap(ix) for ix in np.linspace(0, 1, ns.size).tolist()]
    patches_cpu = [mpatches.Patch(color=color, label=n, linestyle='--', hatch='o') for (color, n) in
                   zip(colormap, [f"CPU n={i}" for i in ns])]

    for i, n in enumerate(ns):
        cpus = df[(df['n'] == n) & (df['device'] == 0)]
        gpus = df[(df['n'] == n) & (df['device'] == 1)]

        cpus.sort_values(by=['t'])
        gpus.sort_values(by=['t'])

        plt.plot(cpus['t'], cpus['fps'], "o--", color=patches_cpu[i].get_facecolor())
        plt.plot(gpus['t'], gpus['fps'], "o--", color=patches_gpu[i].get_facecolor())

    patches = patches_cpu + patches_gpu
    plt.figlegend(handles=patches, labels=list(map(lambda elem: elem.get_label(), patches)), loc='center right',
                  ncol=1,
                  labelspacing=0., borderaxespad=1., title='label')
    plt.xlabel('tamaño de la pantalla t')
    plt.ylabel('fps')
    plt.title('tamaño de la pantalla vs fps')
    plt.savefig('chart.png', dpi=300)


plt.show()
if __name__ == '__main__':
    main()
