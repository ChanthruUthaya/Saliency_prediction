import argparse
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
from scipy import ndimage
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Visualising ground truth saliency maps')

parser.add_argument('--gts', help = 'Ground truth data')
parser.add_argument('--outdir', default = '.', type=Path, help='output directory for visualisation')

args = parser.parse_args()

def main():
    #loading preds and gts
    gts = pickle.load(open(args.gts, 'rb'))

    # index = np.random.randint(0, len(preds), size=3) #get indices for 3 random images
    index = [347]

    outputs = []
    for idx in index:
        #getting original image
        image = gts[idx]['X_original']
        image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
        outputs.append(image)

        #getting ground truth saliency map
        sal_map = gts[idx]['y_original']
        sal_map = ndimage.gaussian_filter(sal_map, 19)
        outputs.append(sal_map)

    #plotting images
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(32,13))
    ax[0].set_title("Image", fontsize=60)
    ax[1].set_title("Saliency Map", fontsize=60)

    fig.tight_layout()

    for i, axi in enumerate(ax.flat):
        axi.imshow(outputs[i])

    #saving output
    if not args.outdir.parent.exists():
        args.outdir.parent.mkdir(parents=True)
    outpath = os.path.join(args.outdir, "gt_vis.jpg")
    plt.savefig(outpath)

if __name__ == '__main__':
    main()
