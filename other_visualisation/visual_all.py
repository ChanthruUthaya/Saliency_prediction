import argparse
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow
from scipy import ndimage
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Visualising model outputs')

parser.add_argument('--preds', help='Model predictions')
parser.add_argument('--gts', help = 'Ground truth data')
parser.add_argument('--outdir', default = 'all_predictions', type=Path, help='output directory for visualisation')

args = parser.parse_args()

def main():
    #loading preds and gts
    preds = pickle.load(open(args.preds, 'rb'))
    gts = pickle.load(open(args.gts, 'rb'))

    # index = np.random.randint(0, len(preds), size=3) #get indices for 3 random images
    img_cluster = 10

    for imgBatch in range(int(500/img_cluster)):
        print(f"working on batch {imgBatch}")
        outputs = []
        for idx in range(imgBatch*img_cluster, imgBatch*img_cluster+10):
            print(f"img {idx}")
            #getting original image
            image = gts[idx]['X_original']
            image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
            outputs.append(image)

            #getting ground truth saliency map
            sal_map = gts[idx]['y_original']
            sal_map = ndimage.gaussian_filter(sal_map, 19)
            outputs.append(sal_map)

            #getting model prediction
            pred = np.reshape(preds[idx], (48, 48))
            pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]))
            pred = np.asarray(pred, dtype='float32') / 255.
            pred = ndimage.gaussian_filter(pred, sigma=2)
            outputs.append(pred)

        #plotting images
        fig, ax = plt.subplots(nrows=img_cluster, ncols=3, figsize=(32,50))
        ax[0][0].set_title("Image", fontsize=40)
        ax[0][1].set_title("Ground Truth", fontsize=40)
        ax[0][2].set_title("Prediction", fontsize=40)

        fig.tight_layout()

        for i, axi in enumerate(ax.flat):
            axi.imshow(outputs[i])

        #saving output
        if not args.outdir.exists():
            args.outdir.mkdir(parents=True)
        outpath = os.path.join(args.outdir, f"output_vis_{imgBatch*img_cluster}-{imgBatch*img_cluster+10}.jpg")
        plt.savefig(outpath)
        plt.close()

if __name__ == '__main__':
    main()
