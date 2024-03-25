import os
import math
import numpy as np
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM

PM_SUFFIX = {"max":"_max", "avg":""}

class FeatureGroupOutput:
    '''A model target for GradCAM that consolidates the outputs of all individual features in a group.
    See ClassifierOutputTarget in pytorch_grad_cam/utils
    '''
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        print(model_output)
        if len(model_output.shape) == 1:
            return model_output[self.features].sum()
        return model_output[:, self.features].sum(dim = 1)


def get_grad_cam(cam, images, targets):
    images.requires_grad = True
    grayscale_cam = cam(input_tensor = images, targets = targets)
    
    return grayscale_cam


def get_clip_image_features2(clip_model, probe_dataset, sample_indices, batch_size = 128, target_feature_group = [], grad_cam = None, resize_transform = None):
    
    images = torch.zeros([sample_indices.shape[0], 3, 224, 224])
    j = 0
    for i in sample_indices.long():
        images[j] = probe_dataset[i.item()][0]
        j += 1
    
    if len(target_feature_group) > 0:
        feat_grad = [FeatureGroupOutput(torch.LongTensor(target_feature_group)) for i in range(images.shape[0])]
    
    cropped_images = []
    clip_image_features = []
    i = 0
    while i * batch_size < images.shape[0]:
        image_batch = images[i * batch_size : (i + 1) * batch_size]

        if grad_cam:
            # 1. Transform with grad cam
            grayscale_mask = get_grad_cam(cam = grad_cam,
                                          images = image_batch,
                                          targets = feat_grad)

            # Crop images to only keep the activated part 
            grayscale_mask = torch.Tensor(grayscale_mask)
            grayscale_mask[grayscale_mask < 0.6] = 0
            grayscale_mask[grayscale_mask >= 0.6] = 1
            
            bb = [torch.LongTensor(mask_to_boxes(m)) for m in grayscale_mask.cpu().numpy()]
            
            image_batch = torch.cat([image_batch[img, : , bb[img][1]:bb[img][3] , bb[img][0]:bb[img][2]].unsqueeze(0) for img in range(image_batch.shape[0])])

            cropped_images.append(image_batch)

        with torch.no_grad():
            clip_image_features.append(clip_model.encode_image(image_batch.cuda()).cuda())

        i += 1
        
    if len(clip_image_features) > 0:
        clip_image_features = torch.cat(clip_image_features, dim = 0)
        clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)  
    else:
        clip_image_features = torch.Tensor([]).cuda()

    cropped_images = images if len(cropped_images) == 0 else torch.cat(cropped_images)
    
    return images, clip_image_features, cropped_images

def mask_to_boxes(mask):
    ''' Convert a boolean (Height x Width) mask into a (N x 4) array of NON-OVERLAPPING bounding boxes
    surrounding 'islands of truth' in the mask.  Boxes indicate the (Left, Top, Right, Bottom) bounds
    of each island, with Right and Bottom being NON-INCLUSIVE (ie they point to the indices AFTER the island).

    This algorithm (Downright Boxing) does not necessarily put separate connected components into
    separate boxes.

    You can 'cut out' the island-masks with
        boxes = mask_to_boxes(mask)
        island_masks = [mask[t:b, l:r] for l, t, r, b in boxes]
    '''
    max_ix = max(s+1 for s in mask.shape)   # Use this to represent background
    # These arrays will be used to carry the 'box start' indices down and to the right.
    x_ixs = np.full(mask.shape, fill_value=max_ix)
    y_ixs = np.full(mask.shape, fill_value=max_ix)

    # Propagate the earliest x-index in each segment to the bottom-right corner of the segment
    for i in range(mask.shape[0]):
        x_fill_ix = max_ix
        for j in range(mask.shape[1]):
            above_cell_ix = x_ixs[i-1, j] if i>0 else max_ix
            still_active = mask[i, j] or ((x_fill_ix != max_ix) and (above_cell_ix != max_ix))
            x_fill_ix = min(x_fill_ix, j, above_cell_ix) if still_active else max_ix
            x_ixs[i, j] = x_fill_ix

    # Propagate the earliest y-index in each segment to the bottom-right corner of the segment
    for j in range(mask.shape[1]):
        y_fill_ix = max_ix
        for i in range(mask.shape[0]):
            left_cell_ix = y_ixs[i, j-1] if j>0 else max_ix
            still_active = mask[i, j] or ((y_fill_ix != max_ix) and (left_cell_ix != max_ix))
            y_fill_ix = min(y_fill_ix, i, left_cell_ix) if still_active else max_ix
            y_ixs[i, j] = y_fill_ix

    # Find the bottom-right corners of each segment
    new_xstops = np.diff((x_ixs != max_ix).astype(np.int32), axis=1, append=False)==-1
    new_ystops = np.diff((y_ixs != max_ix).astype(np.int32), axis=0, append=False)==-1
    corner_mask = new_xstops & new_ystops
    y_stops, x_stops = np.array(np.nonzero(corner_mask))

    # Extract the boxes, getting the top-right corners from the index arrays
    x_starts = x_ixs[y_stops, x_stops]
    y_starts = y_ixs[y_stops, x_stops]
    ltrb_boxes = np.hstack([x_starts[:, None], y_starts[:, None], x_stops[:, None]+1, y_stops[:, None]+1])
    
    max_area_box = None
    max_area = 0
    for bx in ltrb_boxes:
        ar = (bx[2] - bx[0]) * (bx[3] - bx[1])
        if ar > max_area:
            max_area = ar
            max_area_box = bx
        
    return max_area_box

def soft_wpmi(clip_feats, target_feats, top_k=100, a=10, lam=1, device='cuda',
              min_prob=1e-7, p_start=0.998, p_end=0.97):
    with torch.no_grad():
        torch.cuda.empty_cache()
        clip_feats = torch.nn.functional.softmax(a * clip_feats, dim=1)

        top_k = 20
        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        p_in_examples = p_start - (torch.arange(start=0, end=top_k) / top_k * (p_start - p_end)).unsqueeze(1).to(device)
        for orig_id in tqdm(range(target_feats.shape[1])):
            curr_clip_feats = clip_feats.gather(0, inds[:, orig_id:orig_id + 1].expand(-1, clip_feats.shape[1])).to(
                device)

            curr_p_d_given_e = 1 + p_in_examples * (curr_clip_feats - 1)
            curr_p_d_given_e = torch.sum(torch.log(curr_p_d_given_e + min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)
            torch.cuda.empty_cache()

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        print(prob_d_given_e.shape)
        # logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) -
                  torch.log(prob_d_given_e.shape[0] * torch.ones([1]).to(device)))
        mutual_info = prob_d_given_e - lam * prob_d
    return mutual_info


def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                             PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    # if _all_saved(save_names):
    #     return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)): #, shuffle=False)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True, shuffle=False)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features

def get_clip_image_features(model, dataset, device = "cuda"):
    all_features= []

    with torch.no_grad():
        for k, data in enumerate(dataset.val_train_clip_loader):
            images, _, _= data
            features = model.encode_image(images.to(device))
            all_features.append(features)
    img_features = torch.cat(all_features)
    return img_features

def get_target_activations(target_model, dataset, target_layers = ["layer4"], device = "cuda", pool_mode='avg'):
    
    all_features = {target_layer:[] for target_layer in target_layers}

    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)

    with torch.no_grad():
        # for k, train_loader in enumerate(dataset.train_xai_loader):
            # if k < len(dataset.train_loader) - 1:
            #     continue
        for k, data in enumerate(dataset.val_train_xai_loader):
            images, _, _ = data
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        target_features = torch.cat(all_features[target_layer])
        hooks[target_layer].remove()

    return target_features

def caption_images(image_emb, caption_dataloader, caption_dataset):
    top_captions_values = torch.zeros((image_emb.shape[0], 5)).half().cuda()
    top_captions_indices = torch.zeros((image_emb.shape[0], 5)).long().cuda()

    size = 0
    
    for i, (text_emb) in enumerate(caption_dataloader):
        text_emb = text_emb.view(-1, text_emb.shape[-1]).cuda()
        values, indices = (100.0 * image_emb @ text_emb.T).softmax(dim=-1).topk(5)

        cat_values = torch.cat([top_captions_values, values], dim = 1)
        cat_indices = torch.cat([top_captions_indices, indices + size], dim = 1)

        values, indices = cat_values.topk(5)
        top_captions_values = values
        top_captions_indices = torch.cat([cat_indices[i, indices[i]].unsqueeze(0) for i in range(cat_indices.shape[0])])

        size += text_emb.shape[0]
        i += 1
        
    print('SIZE:', size)

    caption_set = []
    score_set = []
    for i in range(top_captions_indices.shape[0]):
        caption_set.append(caption_dataset.get_captions(top_captions_indices[i].cpu().numpy().tolist()))
        score_set.append([val.item() for val in top_captions_values[i]])
        
    return caption_set, score_set


def get_similarity_new(clip_model, target_model, target_layers,
                     concept_set, batch_size, pool_mode, dataset, similarity_type, task=0,
                                   return_target_feats=True, device="cuda"):
    
    
    return_nodes = {
        "layer4": "layer4",
        "avg_pool": "avgpool",
    }
    
    encoder = create_feature_extractor(target_model, return_nodes)
    
    size = len(dataset.val_train_clip_loader.dataset)
    representations = torch.zeros((size, 512))
    for i, (images, _, _) in enumerate(dataset.val_train_clip_loader):
        with torch.no_grad():
            #print(encoder(images.cuda())['layer4'].shape, size)
            representations[i * batch_size: (i + 1) * batch_size] = encoder(images.cuda())['avgpool'].view(-1, 512)
    
    representations = F.normalize(representations, dim = 1)
    
    # Get highly activating sample indices for each feature
    lim = 5 # This limit is empirically selected
    high_thresh = representations.mean() + lim * representations.std()
    print(high_thresh)

    sample_to_rep = {} # feature index to sample indices

    for i in range(representations.shape[1]):
        highly_idx = torch.where(representations[:,i] >= high_thresh)[0]
        if highly_idx.shape[0] > 0:
            for idx in highly_idx:
                if idx.item() in sample_to_rep:
                    sample_to_rep[idx.item()].append(i)
                else:    
                    sample_to_rep[idx.item()] = [i]

    groups = {}
    for sam in sample_to_rep:
        s = ','.join([str(i) for i in sample_to_rep[sam]])
        if s in groups:
            groups[s].append(sam)
        else:
            groups[s] = [sam]
            
    # Prepare grad cam and CLIP to extract interpretable feature groups
    cam = GradCAM(model = encoder,
                  target_layers = [encoder.layer4[-1]] # change target_layers according to the layer we wish to explain
                  ) 
    
    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
        #ignore empty lines
    words = [i for i in words if i!=""]
    
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    text_features = get_clip_text_features(clip_model, text, batch_size)
    
    with torch.no_grad():
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    group_captions = {}
    
    for feat_str in groups:
        feats = [int(i) for i in feat_str.split(',')]
                
        highly_idx = torch.LongTensor(groups[feat_str])

        highly_act_images, highly_act_clip_features, highly_act_cropped_images = get_clip_image_features2(clip_model = clip_model,
                                                                                                            probe_dataset = dataset.val_train_clip_loader.dataset,
                                                                                                            sample_indices = highly_idx,
                                                                                                            batch_size = batch_size,
                                                                                                            target_feature_group = feats,
                                                                                                            grad_cam = cam)
        
        #caption_set, score_set = caption_images(highly_act_clip_features.to(torch.float16), caption_dataloader, caption_dataset)
        values, indices = (100.0 * highly_act_clip_features @ text_features.T).softmax(dim=-1).topk(5)
        
        group_captions[feat_str] = indices[0].cpu().numpy()

    del highly_act_clip_features, text_features
    torch.cuda.empty_cache()
    
    del values, indices
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return group_captions, groups
    else:
        del target_feats
        torch.cuda.empty_cache()
        return group_captions, groups

def get_similarity(clip_model, target_model, target_layers,
                     concept_set, batch_size, pool_mode, dataset, similarity_type, task=0,
                                   return_target_feats=True, device="cuda"):
    
    # clip_model, _ = clip.load(clip_name, device=device)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    #ignore empty lines
    words = [i for i in words if i!=""]
    
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    text_features = get_clip_text_features(clip_model, text, batch_size)
    image_features = get_clip_image_features(clip_model, dataset, device)
    target_feats = get_target_activations(target_model, dataset, target_layers, device, pool_mode)
    
    #image_features = torch.load(clip_save_name, map_location='cpu').float()
    #text_features = torch.load(text_save_name, map_location='cpu').float()
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        clip_feats = (image_features @ text_features.T)

    del image_features, text_features
    torch.cuda.empty_cache()
    
    #target_feats = torch.load(target_save_name, map_location='cpu')
    # similarity_fn = eval("similarity.{}".format(similarity_type))
    similarity = soft_wpmi(clip_feats, target_feats, device=device)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity

def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

    
    