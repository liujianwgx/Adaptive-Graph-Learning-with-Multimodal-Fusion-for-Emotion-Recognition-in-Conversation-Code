import argparse
from numpy.lib.twodim_base import diag

from tqdm import tqdm
import pickle
import os
import json
import pandas as pd
import numpy as np

import cogmen



def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    # for ind, k in enumerate(dataset["vision"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset

def make_dict_old(ids,data):
    items_list=[]
    for idx,value in enumerate(ids):
        items_list.append({value:data[idx]})

    result_dict={}
    for d in items_list:
        for key,value in d.items():
            if key not in result_dict:
                result_dict[key]=[]
            result_dict[key].append(value)
    return result_dict


def make_dict(ids,data,is_str=False):
    items_dict={}
    for idx,value in enumerate(ids):
        items_dict[value]=data[idx]

    result_dict={}

    for key,value in items_dict.items():
        prefix,number=key.split("$_$")
        number=int(number)
        if prefix not in result_dict:
            result_dict[prefix]=[]
        if is_str:
            result_dict[prefix].append((number,str(value)))
        else:
            result_dict[prefix].append((number,value))

    for prefix in result_dict:
        result_dict[prefix]=[item[1] for item in sorted(result_dict[prefix])]

    return result_dict


def main():
    mosei_path="/home/jiawei/COGMEN/emotion_recognition/data/mosei/aligned_50.pkl"
    alldata=pickle.load(open(mosei_path,"rb"))
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid']) ##['vision', 'audio', 'text', 'labels', 'id']
    alldata['test'] = drop_entry(alldata['test'])

    train_video_audio=alldata['train']["audio"]#(16326, 50, 74)
    valid_video_audio=alldata['valid']["audio"]
    test_video_audio=alldata['test']["audio"]
    train_video_visual=alldata['train']["vision"]#(16326, 50, 35)
    valid_video_visual=alldata['valid']["vision"]
    test_video_visual=alldata['test']["vision"]
    train_video_text=alldata['train']["text"]#(16326, 50, 300)
    valid_video_text=alldata['valid']["text"]
    test_video_text=alldata['test']["text"]
    train_video_labels=alldata['train']["regression_labels"]#(16326,)
    valid_video_labels=alldata['valid']["regression_labels"]
    test_video_labels=alldata['test']["regression_labels"]
    train_video_sentence=alldata['train']["raw_text"]#(16326,)
    valid_video_sentence=alldata['valid']["raw_text"]
    test_video_sentence=alldata['test']["raw_text"]
    #old
    # train_video_ids = [row[0] for row in alldata['train']["id"]]
    # valid_video_ids = [row[0] for row in alldata['valid']["id"]]
    # test_video_ids = [row[0] for row in alldata['test']["id"]]

    train_video_ids = alldata['train']["id"]
    valid_video_ids = alldata['valid']["id"]
    test_video_ids = alldata['test']["id"]
    # train_vids=list(dict.fromkeys(train_video_ids))
    # valid_vids=list(dict.fromkeys(valid_video_ids))
    # test_vids=list(dict.fromkeys(test_video_ids))

    train_audio={}
    train_text={}
    train_visual={}
    train_labels={}
    valid_audio={}
    valid_text={}
    valid_visual={}
    valid_labels={}
    test_audio={}
    test_text={}
    test_visual={}
    test_labels={}



    train_audio=make_dict(train_video_ids,train_video_audio)
    train_text=make_dict(train_video_ids,train_video_text)
    train_visual=make_dict(train_video_ids,train_video_visual)
    train_labels=make_dict(train_video_ids,train_video_labels)
    train_sentence=make_dict(train_video_ids,train_video_sentence,is_str=True)

    valid_audio=make_dict(valid_video_ids,valid_video_audio)
    valid_text=make_dict(valid_video_ids,valid_video_text)
    valid_visual=make_dict(valid_video_ids,valid_video_visual)
    valid_labels=make_dict(valid_video_ids,valid_video_labels)
    valid_sentence=make_dict(valid_video_ids,valid_video_sentence,is_str=True)

    test_audio=make_dict(test_video_ids,test_video_audio)
    test_text=make_dict(test_video_ids,test_video_text)
    test_visual=make_dict(test_video_ids,test_video_visual)
    test_labels=make_dict(test_video_ids,test_video_labels)
    test_sentence=make_dict(test_video_ids,test_video_sentence,is_str=True)

    train_vids=list(train_audio.keys())
    valid_vids=list(valid_audio.keys())
    test_vids=list(test_audio.keys())

    # for idx,value in enumerate(train_video_ids):
    #     train_audio_items.append({value:train_video_audio[idx]})
    #     train_text_items.append({value:train_video_text[idx]})
    #     train_visual_items.append({value:train_video_visual[idx]})
    #     train_labels_items.append({value:train_video_labels[idx]})

    # for d in train_audio_items:
    #     for key,value in d.items():
    #         if key not in train_audio:
    #             train_audio[key]=[]
    #         train_audio[key].append(value)
    data={
        "train":{"labels":train_labels,"text":train_text,"audio":train_audio,"visual":train_visual,"sentence":train_sentence},
        "valid":{"labels":valid_labels,"text":valid_text,"audio":valid_audio,"visual":valid_visual,"sentence":valid_sentence},
        "test":{"labels":test_labels,"text":test_text,"audio":test_audio,"visual":test_visual,"sentence":test_sentence},
        "train_vids":train_vids,"valid_vids":valid_vids,"test_vids":test_vids
    }
    cogmen.utils.save_pkl(data, "/home/jiawei/COGMEN/emotion_recognition/data/mosei/mosei_prepare_aligned_data.pkl")
    
    
    

if __name__ == '__main__':
    main()