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
parser.add_argument('--outdir', default = '.', type=Path, help='output directory for visualisation')

args = parser.parse_args()

def main():
    #loading preds and gts
    preds = pickle.load(open(args.preds, 'rb'))
    gts = pickle.load(open(args.gts, 'rb'))

    # index = np.random.randint(0, len(preds), size=3) #get indices for 3 random images
    index = [209, 8, 138]

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

        #getting model prediction
        pred = np.reshape(preds[idx], (48, 48))
        # print(f"min of original pred {np.min(preds)}")
        # print(f"min before uint {np.min(preds*255)}")
        # numpy.unravel_index(A.argmin(), A.shape)
        # print(f"smallest coordinate {np.unravel_index((preds*255).argmin(), (preds*255).shape)}")
        # temp = np.unravel_index((preds*255).argmin(), (preds*255).shape)
        # temp2 = np.unravel_index((preds*255).astype(np.uint8).argmin(), (preds*255).astype(np.uint8).shape)
        # print((preds*255)[temp[0], temp[1]])
        # print((preds*255).astype(np.uint8)[temp[0], temp[1]])
        # print(f"min after uint {np.min((preds*255).astype(np.uint8))}")
        # print(f"smallest coordinate after uint {np.argmin((preds*255).astype(np.uint8))}")
        # print(f"img shape is {image.shape}, max of pred is {np.max((pred*255).astype(np.uint8))}, argmax is {np.argmax(pred*255)}")
        no_uint = Image.fromarray((pred*255)).resize((image.shape[1], image.shape[0]))
        with_uint = Image.fromarray((pred*255).astype(np.uint8)).resize((image.shape[1], image.shape[0]))

        # print((np.asarray(no_uint, dtype='float32') / 255.)[400,500])
        # print((np.asarray(with_uint, dtype='float32') / 255.)[400,500])
        # temp = np.unravel_index(np.argmin(no_uint), image.shape)
        # temp2 = np.unravel_index(np.argmin(with_uint), image.shape)
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        outputs.append(pred)

    #plotting images
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(32,32))
    ax[0][0].set_title("Image", fontsize=60)
    ax[0][1].set_title("Ground Truth", fontsize=60)
    ax[0][2].set_title("Prediction", fontsize=60)

    fig.tight_layout()

    for i, axi in enumerate(ax.flat):
        # circle1 = plt.Circle((500,400), 30, color='r')
        # print(temp2[0], temp2[1])
        # circle2 = plt.Circle((temp2[0], temp2[1]), 30, color='b')
        axi.imshow(outputs[i])
        # axi.add_artist(circle1)
        # axi.add_artist(circle2)
        # if i == 2:
        #     break

    #saving output
    if not args.outdir.parent.exists():
        args.outdir.parent.mkdir(parents=True)
    outpath = os.path.join(args.outdir, "selected_vis.jpg")
    plt.savefig(outpath)

if __name__ == '__main__':
    main()
