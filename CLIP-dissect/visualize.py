import os
import argparse
import random
import numpy as np
import json
import pandas as pd
import torch

import utils
import similarity
import data_utils

parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16", 
                    choices=['RN18','RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default="resnet50", 
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target_path", type=str, default=" ",
                   help="path to target model")
parser.add_argument("--target_layers", type=str, default="conv1,layer1,layer2,layer3,layer4",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--task", type=int, default=0, help="task of cl")

parser.add_argument("--d_probe", type=str, default="broden", 
                    choices = ["imagenet_broden", "cifar100_val", "cifar10_val", "imagenet_val", "broden", "imagenet_broden"])
parser.add_argument("--concept_set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder", 
                                                                               "cos_similarity", "cos_similarity_cubed"])

parser.parse_args()

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parser.parse_args()
    args.target_layers = args.target_layers.split(",")

    set_random_seed(0)

    similarity_fn = eval("similarity.{}".format(args.similarity_fn))
    
    utils.save_activations(clip_name = args.clip_model, target_name = args.target_model, 
                           target_layers = args.target_layers, d_probe = args.d_probe, 
                           concept_set = args.concept_set, batch_size = args.batch_size, 
                           device = args.device, pool_mode=args.pool_mode, 
                           save_dir = args.activation_dir, path=args.target_path, task=args.task)
    
    outputs = {"sim_id":[], "similarity_fn": [],"model":[], "layer":[], "unit":[], "description":[], "similarity":[]}
    model_suffix = os.path.splitext(os.path.basename(args.target_path))[0]
    outputs["model"] = model_suffix
    outputs["similarity_fn"] = args.similarity_fn

    with open(args.concept_set, 'r') as f: 
        words = (f.read()).split('\n')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    save_path = "{}/{}_{}_{}".format(args.result_dir, args.target_model, model_suffix, args.target_layers) #, datetime.datetime.now().strftime("%y_%m_%d_%H_%M"))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    for target_layer in args.target_layers:
        save_names = utils.get_save_names(clip_name = args.clip_model, target_name = args.target_model,
                                  target_layer = target_layer, d_probe = args.d_probe,
                                  concept_set = args.concept_set, pool_mode = args.pool_mode,
                                  save_dir = args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names

        similarities, target_feats = utils.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=True, device=args.device
        )

        pil_data = data_utils.get_data(args.d_probe, task=args.task)
        # data_utils.visualize(args, target_feats, similarities, words, pil_data, save_path)

        vals, ids = torch.max(similarities, dim=1)
        descriptions = [words[int(idx)] for idx in ids]
        i =0
        outputs["sim_id"].extend([i] * len(vals))
        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer] * len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())

        # vals_all, ids_all = torch.topk(similarities, k=similarities.shape[1], largest=True)
        # for i in range(vals_all.size(1)):
        #     # del similarities
        #     torch.cuda.empty_cache()
        #     vals = vals_all[:,i]
        #     ids = ids_all[:,i]
        #     descriptions = [words[int(idx)] for idx in ids]
        #     outputs["sim_id"].extend([i] * len(vals))
        #     outputs["unit"].extend([i for i in range(len(vals))])
        #     outputs["layer"].extend([target_layer]*len(vals))
        #     outputs["description"].extend(descriptions)
        #     outputs["similarity"].extend(vals.cpu().numpy())

    df = pd.DataFrame(outputs)
    df.to_csv(os.path.join(save_path,"descriptions.csv"), index=False)
    with open(os.path.join(save_path, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)