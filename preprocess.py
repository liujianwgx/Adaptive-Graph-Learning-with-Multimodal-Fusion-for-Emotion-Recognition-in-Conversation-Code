import argparse
from numpy.lib.twodim_base import diag

from tqdm import tqdm
import pickle
import os
import json
import pandas as pd
import numpy as np

import cogmen
from get_data import get_dataloader
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

log = cogmen.utils.get_logger()

def get_iemocap():
    cogmen.utils.set_seed(args.seed)

    if args.dataset == "iemocap":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("/home/jiawei/COGMEN/emotion_recognition/data/iemocap/IEMOCAP_features.pkl", "rb"), encoding="latin1"
        )
    elif args.dataset == "iemocap_4":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/iemocap_4/IEMOCAP_features_4.pkl", "rb"), encoding="latin1"
        )

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            cogmen.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            cogmen.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            cogmen.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    # all_labels=list(video_labels.values())
    # emotions=list(set(label for session in all_labels for label in session))
    # emotion_to_idx={emotion:idx for idx,emotion in enumerate(emotions)}
    # transition_matrix = np.zeros((len(emotions), len(emotions)))

    # categories=["happiness","sadness","neutral","anger","excited","frustrated"]

    # # 计算情感转移频次
    # for session in all_labels:
    #     for i in range(len(session) - 1):
    #         current_emotion = session[i]
    #         next_emotion = session[i + 1]
    #         transition_matrix[emotion_to_idx[current_emotion], emotion_to_idx[next_emotion]] += 1

    # # 绘制情感转移矩阵
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(transition_matrix, annot=True, fmt="g", cmap="Blues",
    #             xticklabels=categories, yticklabels=categories)
    # plt.xlabel('Transition from', fontsize=12)
    # plt.ylabel('Transition to', fontsize=12)
    # plt.show()

    
    # categories=list(label_counts.keys())

    # all_labels=[]
    # for labels in video_labels.values():
    #     all_labels.extend(labels)
    #     # 使用 Counter 统计频率
    # label_counts = Counter(all_labels)

    # # 准备数据用于绘制条形图
    # categories = list(label_counts.keys())
    # # categories=["happiness","sadness","neutral","anger","excited","frustrated"]
    # dataset_label_dict = {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5}
    # label_to_category = {v: k for k, v in dataset_label_dict.items()}
    # english_categories = [label_to_category[label] for label in categories]
    # counts = list(label_counts.values())

    # # 绘制条形图
    # plt.figure(figsize=(10, 6))
    # plt.bar(english_categories, counts, color='skyblue')
    # plt.xlabel('type', fontsize=12)
    # plt.ylabel('count', fontsize=12)
    # plt.title('每个类别的出现次数', fontsize=14)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # 显示图形
    # plt.show()





    return train, dev, test


def get_iemocap_split(split_utterances):
    cogmen.utils.set_seed(args.seed)

    if args.dataset == "iemocap":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/iemocap/IEMOCAP_features.pkl", "rb"), encoding="latin1"
        )
    elif args.dataset == "iemocap_4":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/iemocap_4/IEMOCAP_features_4.pkl", "rb"), encoding="latin1"
        )

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            train.append(
                cogmen.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    for vid in tqdm(dev_vids, desc="dev"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            dev.append(
                cogmen.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    for vid in tqdm(test_vids, desc="test"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            test.append(
                cogmen.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def get_mosei_from_tbje(args):
    def cmumosei_7(a):
        if a < -2:
            res = 0
        if -2 <= a and a < -1:
            res = 1
        if -1 <= a and a < 0:
            res = 2
        if 0 <= a and a <= 0:
            res = 3
        if 0 < a and a <= 1:
            res = 4
        if 1 < a and a <= 2:
            res = 5
        if a > 2:
            res = 6
        return res

    def cmumosei_2(a):
        if a < 0:
            return 0
        if a >= 0:
            return 1

    (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        trainVids,
        dev_vids,
        test_vids,
    ) = pickle.load(
        open("./data/mosei/tbje_mosei_updated.pkl", "rb"), encoding="latin1"
    )

    # alldata=pickle.load(open("/home/jiawei/COGMEN/emotion_recognition/data/MOSEI/mosei_senti_data.pkl","rb"))
    # alldata['train'] = drop_entry(alldata['train'])
    # alldata['valid'] = drop_entry(alldata['valid'])
    # alldata['test'] = drop_entry(alldata['test'])

    train, dev, test = [], [], []

    video_ids = np.array(list(video_ids.items()))

    for dialogue_idx in tqdm(trainVids, desc="train"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]
        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei_tbje_7class":
            labels = [
                cmumosei_7(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_tbje_2class":
            labels = [
                cmumosei_2(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        train.append(
            cogmen.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(dev_vids, desc="dev"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            ()

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei_tbje_7class":
            labels = [
                cmumosei_7(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_tbje_2class":
            labels = [
                cmumosei_2(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]

        dev.append(
            cogmen.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(test_vids, desc="test"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei_tbje_7class":
            labels = [
                cmumosei_7(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_tbje_2class":
            labels = [
                cmumosei_2(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]

        test.append(
            cogmen.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    return train, dev, test


def get_mosei_from_tbje_emotion(args):
    (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        trainVids,
        dev_vids,
        test_vids,
    ) = pickle.load(
        open("./data/mosei/tbje_mosei_" + args.emotion + ".pkl", "rb"),
        encoding="latin1",
    )

    train, dev, test = [], [], []

    video_ids = np.array(list(video_ids.items()))

    for dialogue_idx in tqdm(trainVids, desc="train"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]
        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        labels = [video_labels[dialogue_idx][i] for i in range(num_of_utterances)]
        train.append(
            cogmen.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(dev_vids, desc="dev"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        labels = [video_labels[dialogue_idx][i] for i in range(num_of_utterances)]

        dev.append(
            cogmen.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(test_vids, desc="test"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        labels = [video_labels[dialogue_idx][i] for i in range(num_of_utterances)]

        test.append(
            cogmen.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    return train, dev, test


def get_mosei():

    cogmen.utils.set_seed(args.seed)

    def cmumosei_7(a):
        if a < -2:
            res = 0
        if -2 <= a and a < -1:
            res = 1
        if -1 <= a and a < 0:
            res = 2
        if 0 <= a and a <= 0:
            res = 3
        if 0 < a and a <= 1:
            res = 4
        if 1 < a and a <= 2:
            res = 5
        if a > 2:
            res = 6
        return res

    def cmumosei_2(a):
        if a < 0:
            return 0
        if a >= 0:
            return 1

    # feature_path = "categorical.p"
    # path = os.path.join(mosei_path, feature_path)
    # (
    #     video_ids,
    #     video_speakers,
    #     video_labels,
    #     video_text,
    #     video_audio,
    #     video_visual,
    #     video_sentence,
    #     trainVids,
    #     test_vids,
    # ) = pickle.load(open(path, "rb"), encoding="latin1")
    alldata=pickle.load(open("/home/jiawei/COGMEN/emotion_recognition/data/mosei/mosei_prepare_aligned_data.pkl","rb"))
    # train_dataset=alldata["train"]
    # valid_dataset=alldata["valid"]
    # test_dataset=alldata["test"]

    train_labels=alldata["train"]["labels"]
    train_text=alldata["train"]["text"]#dim:768
    train_audio=alldata["train"]["audio"]#dim:74
    train_visual=alldata["train"]["visual"]#dim:35
    train_sentence=alldata["train"]["sentence"]
    valid_labels=alldata["valid"]["labels"]
    valid_text=alldata["valid"]["text"]
    valid_audio=alldata["valid"]["audio"]
    valid_visual=alldata["valid"]["visual"]
    valid_sentence=alldata["valid"]["sentence"]
    test_labels=alldata["test"]["labels"]
    test_text=alldata["test"]["text"]
    test_audio=alldata["test"]["audio"]
    test_visual=alldata["test"]["visual"]
    test_sentence=alldata["test"]["sentence"]
    train_vids=alldata["train_vids"]
    valid_vids=alldata["valid_vids"]
    test_vids=alldata["test_vids"]

    train, dev, test = [], [], []

    for dialogue_idx in tqdm(train_vids,desc="train"):
        num_of_utterances = len(train_audio[dialogue_idx])
        audio = [
            np.average(train_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##audio dim=74
        # text = [" ".join(i) for i in train_text[dialogue_idx]]
        text = [
            np.average(train_text[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##text dim=300
        visual = [
            np.average(train_visual[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##visual dim=35

        sentence=train_sentence[dialogue_idx]

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei":
            labels = [
                cmumosei_7(train_labels[dialogue_idx][i])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_2":
            labels = [
                cmumosei_2(train_labels[dialogue_idx][i])
                for i in range(num_of_utterances)
            ]
        train.append(
            cogmen.Sample(
                str(dialogue_idx), speakers, labels, text, audio, visual, sentence
            )
        )

    for dialogue_idx in tqdm(valid_vids,desc="dev"):
        num_of_utterances = len(valid_audio[dialogue_idx])
        audio = [
            np.average(valid_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##audio dim=74
        # text = [" ".join(i) for i in train_text[dialogue_idx]]
        text = [
            np.average(valid_text[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##text dim=300
        visual = [
            np.average(valid_visual[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##visual dim=35
        sentence=valid_sentence[dialogue_idx]

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei":
            labels = [
                cmumosei_7(valid_labels[dialogue_idx][i])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_2":
            labels = [
                cmumosei_2(valid_labels[dialogue_idx][i])
                for i in range(num_of_utterances)
            ]
        dev.append(
            cogmen.Sample(
                str(dialogue_idx), speakers, labels, text, audio, visual, sentence
            )
        )

    for dialogue_idx in tqdm(test_vids,desc="test"):
        num_of_utterances = len(test_audio[dialogue_idx])
        audio = [
            np.average(test_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##audio dim=74
        # text = [" ".join(i) for i in train_text[dialogue_idx]]
        text = [
            np.average(test_text[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##text dim=300
        visual = [
            np.average(test_visual[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]##visual dim=35
        sentence=test_sentence[dialogue_idx]

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei":
            labels = [
                cmumosei_7(test_labels[dialogue_idx][i])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_2":
            labels = [
                cmumosei_2(test_labels[dialogue_idx][i])
                for i in range(num_of_utterances)
            ]
        test.append(
            cogmen.Sample(
                str(dialogue_idx), speakers, labels, text, audio, visual, sentence 
            )
        )

    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(valid_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def main(args):
    if args.dataset == "iemocap":
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        # cogmen.utils.save_pkl(data, "./data/iemocap/data_iemocap.pkl")
    if args.dataset == "iemocap_4" and args.split_utterances == -1:
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        # cogmen.utils.save_pkl(data, "./data/iemocap_4/data_iemocap_4.pkl")
    if args.dataset == "iemocap_4" and args.split_utterances != -1:
        train, dev, test = get_iemocap_split(args.split_utterances)
        data = {"train": train, "dev": dev, "test": test}
        cogmen.utils.save_pkl(
            data,
            "./data/iemocap_4/data_iemocap_4_split_"
            + str(args.split_utterances)
            + ".pkl",
        )
    if args.dataset == "mosei":
        train, dev, test = get_mosei()
        data = {"train": train, "dev": dev, "test": test}
        cogmen.utils.save_pkl(data, "./data/mosei/data_mosei.pkl")
    if args.dataset == "mosei_2":
        train, dev, test = get_mosei()
        data = {"train": train, "dev": dev, "test": test}
        cogmen.utils.save_pkl(data, "./data/mosei/data_mosei_2.pkl")
    # if args.dataset == "mosei_tbje_2class":
    #     train, dev, test = get_mosei_from_tbje(args)
    #     data = {"train": train, "dev": dev, "test": test}
    #     cogmen.utils.save_pkl(data, "./data/mosei/data_mosei_2class.pkl")

    # if args.dataset == "mosei_tbje_7class":
    #     train, dev, test = get_mosei_from_tbje(args)
    #     data = {"train": train, "dev": dev, "test": test}
    #     cogmen.utils.save_pkl(data, "./data/mosei/data_mosei_7class.pkl")

    # if args.dataset == "mosei_emotion":
    #     train, dev, test = get_mosei_from_tbje_emotion(args)
    #     data = {"train": train, "dev": dev, "test": test}
    #     cogmen.utils.save_pkl(data, "./data/mosei/data_mosei_" + args.emotion + ".pkl")

    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument(
        "--dataset",
        type=str,
        default="iemocap",
        choices=["iemocap", "iemocap_4", "mosei","mosei_2",],
        help="Dataset name.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Dataset directory"
    )
    parser.add_argument(
        "--use_wave2vec2_audio_features",
        action="store_true",
        default=False,
        help="uses wave2vec2 extracted audio features",
    )
    parser.add_argument(
        "--use_pose_visual_features",
        action="store_true",
        default=False,
        help="uses extracted pose from visual modality",
    )
    parser.add_argument("--split_utterances", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    args = parser.parse_args()

    main(args)
