import pickle
import os
import argparse
import torch
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from tqdm import tqdm
from thop import profile

import cogmen

log = cogmen.utils.get_logger()


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def warm_up(model,dataset,device):
    data=dataset[0]
    for k, v in data.items():
        if not k == "utterance_texts":
            data[k] = v.to(device)

    with torch.no_grad():
        for i in range(10):
            _=model(data)
    flops,params=profile(model,(data,))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

def draw_confusion_matrix(golds,preds,label_to_idx):
    cm=metrics.confusion_matrix(golds,preds)
    fig,ax=plt.subplots()
    plt.xlabel('Predicted label',fontsize=12)
    plt.ylabel('True label',fontsize=12)
    im=ax.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    ax.figure.colorbar(im,ax=ax)
    ax.set(xticks=np.arange(cm.shape[0]),yticks=np.arange(cm.shape[1]),
    xticklabels=label_to_idx.keys(),
    yticklabels=label_to_idx.keys())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>0.5*np.max(cm) else "black")
    fig.tight_layout()
    plt.show()

  


def main(args):
    data = load_pkl(f"data/{args.dataset}/data_{args.dataset}.pkl")

    model_dict = torch.load(
        "model_checkpoints/"
        + str(args.dataset)
        + "_best_dev_f1_model_"
        + str(args.modalities)
        + ".pt",
    )
    stored_args = model_dict["args"]
    model = model_dict["state_dict"]
    testset = cogmen.Dataset(data["test"], stored_args)

    model.to(stored_args.device)

    starter,ender=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

    warm_up(model,testset,stored_args.device)

    test = True

    dataset_label_dict = {
        "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
        "mosei": {
            "Strong Negative": 0,
            "Negative": 1,
            "Weak Negative": 2,
            "Neutral": 3,
            "Weak Positive": 4,
            "Positive": 5,
            "Strong Positive": 6,
        },
        "mosei_2": {"Negative": 0, "Positive": 1},
    }
    label_to_idx = dataset_label_dict[args.dataset]
    with torch.no_grad():
        golds = []
        preds = []
        timings=[]
        for idx in tqdm(range(len(testset)), desc="test" if test else "dev"):
            data = testset[idx]
            golds.append(data["label_tensor"])
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(stored_args.device)
            #记录推理时间
            starter.record()
            # nll = model.get_loss(data)
            y_hat = model(data)
            ender.record()
            torch.cuda.synchronize()
            curr_time=starter.elapsed_time(ender)
            timings.append(curr_time)

            preds.append(y_hat.detach().to("cpu"))

        mean_time=np.mean(np.array(timings))
        print("Mean Inference Time: {:.4f} ms".format(mean_time))
   

        if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
        else:
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

        if test:
            print(metrics.classification_report(golds, preds, target_names=label_to_idx.keys(),digits=4))
            draw_confusion_matrix(golds,preds,label_to_idx)


            if stored_args.dataset == "mosei" and stored_args.emotion == "multilabel":
                happy = metrics.f1_score(golds[:, 0], preds[:, 0], average="weighted")
                sad = metrics.f1_score(golds[:, 1], preds[:, 1], average="weighted")
                anger = metrics.f1_score(golds[:, 2], preds[:, 2], average="weighted")
                surprise = metrics.f1_score(
                    golds[:, 3], preds[:, 3], average="weighted"
                )
                disgust = metrics.f1_score(golds[:, 4], preds[:, 4], average="weighted")
                fear = metrics.f1_score(golds[:, 5], preds[:, 5], average="weighted")

                f1 = {
                    "happy": happy,
                    "sad": sad,
                    "anger": anger,
                    "surprise": surprise,
                    "disgust": disgust,
                    "fear": fear,
                }

            print(f"F1 Score: {f1}")
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="eval.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei","mosei_2"],
        help="Dataset name.",
    )

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    parser.add_argument("--device", type=str, default="cpu", help="Computing device.")

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "at", "atv"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    parser.add_argument("--visualize",type=bool,default=True,help="show Umap")

    args = parser.parse_args()
    main(args)
