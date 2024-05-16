import logging
from collections import defaultdict

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
import matplotlib.cm as cm
from tqdm.auto import tqdm

from wbfm.utils.neuron_matching.utils_matching import calc_matches_from_positions_using_softmax
from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.tracklets.high_performance_pandas import get_name_mapping_for_track_dataframes
from wbfm.utils.neuron_matching.matches_class import MatchesWithConfidence
from wbfm.utils.nn_utils.worm_with_classifier import WormWithNeuronClassifier
from wbfm.utils.projects.finished_project_data import template_matches_to_dataframe


def test_trained_classifier(dataloader, model,
                            which_dataset='test_dataset', loss_fn=None, device=None):
    if loss_fn is None:
        loss_fn = model.criterion
    if device is None:
        device = model.device
    dataset = dataloader.__getattribute__(which_dataset)
    size = len(dataset)
    num_batches = len(dataset)
    model.eval()
    test_loss, correct = 0, 0

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    with torch.no_grad():
        for X, y in tqdm(dataset, leave=False):
            X, y = X.to(device), torch.unsqueeze(torch.tensor(y), dim=0).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            correct_vec = (pred.argmax(1) == y).type(torch.float)
            correct += correct_vec.sum().item()

            for k, v in zip(y, correct_vec):
                k = int(k.to("cpu").numpy())
                correct_per_class[k] += v.to("cpu").numpy()
                total_per_class[k] += 1
    test_loss /= num_batches
    correct /= size
    out = f"Test Error: \n Accuracy: {(100 *correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    logging.info(out)
    print(out)

    return correct_per_class, total_per_class


def test_trained_embedding_matcher(dataloader, model,
                                   which_dataset='test_dataset',
                                   trained_embedding=None, trained_labels=None,
                                   loss_fn=None):
    if loss_fn is None:
        loss_fn = model.criterion
    if trained_embedding is None:
        trained_labels, trained_embedding = build_template_from_loader(dataloader, model)
    dataset = dataloader.__getattribute__(which_dataset)

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    model.eval()

    with torch.no_grad():
        for i, (volume, labels) in enumerate(tqdm(dataset)):
            volume = torch.squeeze(volume.to(model.device))
            # labels = labels[0].to("cpu").numpy().astype(int)
            # Should I check for binary
            query_embedding = model.embed(volume).astype(float)

            matches = calc_matches_from_positions_using_softmax(query_embedding, trained_embedding)

            num_correct_this_volume = 0
            for i_t, i_q in matches:
                if labels[i_q] != -1:
                    if trained_labels[i_t] == labels[i_q]:
                        correct_per_class[labels[i_q]] += 1
                        num_correct_this_volume += 1
                    total_per_class[labels[i_q]] += 1

    correct = sum(correct_per_class.values()) / sum(total_per_class.values())
    out = f"Test Error: \n Accuracy: {(100 *correct):>0.1f}%\n"
    logging.info(out)
    print(out)

    return correct_per_class, total_per_class


def test_open_set_tracking(project_data, neurons_that_are_finished, model=None, all_frames=None, tracker_opt=None):
    # Build and use tracker class
    if tracker_opt is None:
        tracker_opt = {}
    tracking_cfg = project_data.project_config.get_tracking_config()
    t_template = tracking_cfg.config['final_3d_tracks'].get('template_time_point', 10)

    if all_frames is None:
        all_frames = project_data.raw_frames
    num_frames = len(all_frames)
    tracker = WormWithNeuronClassifier(template_frame=all_frames[t_template], model=model, **tracker_opt)

    all_matches = []
    for t in tqdm(range(num_frames)):
        matches_with_conf = tracker.match_target_frame(all_frames[t])
        all_matches.append(matches_with_conf)
    df_tracker = template_matches_to_dataframe(project_data, all_matches)
    df_gt = project_data.final_tracks

    return test_open_set_tracking_from_dataframe(df_tracker, df_gt, neurons_that_are_finished)


def test_adjacent_time_point_tracking(project_data, model, neurons_that_are_finished):
    # Build and use tracker class
    num_frames = project_data.num_frames
    all_frames = project_data.raw_frames

    all_matches = []
    for t in tqdm(range(num_frames - 1)):
        prev_frame, next_frame = all_frames[t], all_frames[t + 1]
        tracker = WormWithNeuronClassifier(template_frame=prev_frame, model=model)
        matches_with_conf = tracker.match_target_frame(next_frame)
        all_matches.append(matches_with_conf)

    # Cast as better objects
    tmp = [np.array(m) for m in all_matches]
    all_matches_obj = [MatchesWithConfidence.matches_from_array(m) for m in tmp]
    all_matches_dicts = [m.get_mapping_0_to_1(unique=True) for m in all_matches_obj]
    all_conf_dicts = [m.get_mapping_pair_to_conf() for m in all_matches_obj]
    correct_per_class = defaultdict(int)
    accuracy_correct_per_class = defaultdict(list)
    accuracy_incorrect_per_class = defaultdict(list)
    total_per_class = defaultdict(int)

    df_gt = project_data.final_tracks
    for i, gt_neuron_name in enumerate(tqdm(neurons_that_are_finished)):
        gt_neuron_ind = list(df_gt.loc[:, (gt_neuron_name, 'raw_neuron_ind_in_list')])

        prev_gt_ind = cast_int_or_nan(gt_neuron_ind[0])

        for t_minus_1, gt in enumerate(gt_neuron_ind[1:]):
            if np.isnan(gt):
                continue

            gt = int(gt)
            tracker_match = all_matches_dicts[t_minus_1].get(prev_gt_ind, None)
            conf = all_conf_dicts[t_minus_1].get((prev_gt_ind, gt), 0)

            prev_gt_ind = gt

            if tracker_match and gt == tracker_match:
                correct_per_class[gt_neuron_name] += 1
                accuracy_correct_per_class[gt_neuron_name].append(conf)
            else:
                accuracy_incorrect_per_class[gt_neuron_name].append(conf)
            total_per_class[gt_neuron_name] += 1

    return correct_per_class, total_per_class, \
           accuracy_correct_per_class, accuracy_incorrect_per_class


def test_open_set_tracking_from_dataframe(df_tracker, df_gt, neurons_that_are_finished, verbose=0):
    # NOTE: the neuron names as used in the tracker may not match the names as used in the database
    df_new = df_tracker
    correct_per_class = defaultdict(int)
    accuracy_correct_per_class = defaultdict(list)
    accuracy_incorrect_per_class = defaultdict(list)
    total_per_class = defaultdict(int)
    # Use multiple templates, in case the gt has an error on that frame
    templates = [10, 50, 100, 1000, 1010, 1100, 1500, 2000]
    dfold2dfnew_dict, out = get_name_mapping_for_track_dataframes(df_gt, df_new,
                                                                  t_templates=templates,
                                                                  names_old=neurons_that_are_finished)
    # Only the neurons in this list are actually correct in the gt dataframe
    for i, gt_neuron_name in enumerate(neurons_that_are_finished):
        gt_neuron_ind = list(df_gt.loc[:, (gt_neuron_name, 'raw_neuron_ind_in_list')])
        new_neuron_name = dfold2dfnew_dict.get(gt_neuron_name, None)
        if new_neuron_name is None:
            if verbose >= 1:
                print(f"Did not find old name {gt_neuron_name} in the new tracker")
            continue
        new_neuron_ind = list(df_new.loc[:, (new_neuron_name, 'raw_neuron_ind_in_list')])

        for t, (gt, new) in enumerate(zip(gt_neuron_ind, new_neuron_ind)):
            if np.isnan(gt):
                continue
            if gt == new:
                correct_per_class[gt_neuron_name] += 1
                accuracy_correct_per_class[gt_neuron_name].append(df_new.at[t, (new_neuron_name, 'likelihood')])
            else:
                accuracy_incorrect_per_class[gt_neuron_name].append(df_new.at[t, (new_neuron_name, 'likelihood')])
            total_per_class[gt_neuron_name] += 1
    mean_acc = np.mean([cor / tot for cor, tot in zip(correct_per_class.values(), total_per_class.values())])
    print(f"Mean accuracy: {mean_acc}")
    return correct_per_class, total_per_class, dfold2dfnew_dict, \
           accuracy_correct_per_class, accuracy_incorrect_per_class, mean_acc, df_new


def build_template_from_loader(volume_module, model):
    # Get template points corresponding to one volume
    try:
        alldata_loader = volume_module.alldata
    except AttributeError:
        volume_module.setup()
        alldata_loader = volume_module.alldata

    with torch.no_grad():
        trained_template, trained_labels = alldata_loader[0]
        trained_embedding = model.embed(trained_template).astype(float)

    return trained_labels, trained_embedding


def plot_accuracy(correct_per_class=None, total_per_class=None):
    plt.figure(figsize=(25, 15))

    x = list(correct_per_class.keys())
    y = []
    for i in x:
        y.append(correct_per_class[i] / total_per_class[i])

    sns.barplot(x=x, y=y)
    plt.ylabel("Fraction correct")
    plt.xlabel("Neuron")
    plt.title("Accuracy across test set (annotated neurons)")


def embed_all_points(dataloader, model, device="cpu"):
    model.eval()

    all_embeddings = defaultdict(list)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            embed = model.embed(X)

            for k, v in zip(y, embed):
                k = int(k.to("cpu").numpy())
                v = v.to("cpu").numpy()
                all_embeddings[k].append(v)

    for k, v in all_embeddings.items():
        all_embeddings[k] = np.vstack(v)

    return all_embeddings


def tsne_plot_embeddings(all_feature_spaces=None):
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=100,
        n_iter=300,
    )

    all_x_tsne = tsne.fit_transform(np.vstack(all_feature_spaces))

    all_lens = list(map(len, all_feature_spaces))

    colors = cm.nipy_spectral(np.linspace(0, 1, len(all_feature_spaces)))

    plt.figure(figsize=(15, 10))

    last_len = 0
    for i, this_len in enumerate(all_lens):
        this_len += last_len
        plt.scatter(all_x_tsne[last_len:this_len, 0], all_x_tsne[last_len:this_len, 1],
                    c=np.expand_dims(colors[i], axis=0))
        last_len = this_len
