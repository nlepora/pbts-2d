import os, json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath # for Unix compatibility
from pose_models_2d.lib.models.cnn_model import CNNmodel

data_path = os.path.join(os.environ["DATAPATH"], "open", "tactile-servoing-2d-dobot")


def make_meta(meta_file, model_meta_file):
    # User-defined paths
    home_dir = os.path.join("digitac", "model_edge2d")
    model_dir = os.path.join(home_dir, "train", "train2d_cnn_opt")
    data_dir = os.path.join(home_dir, "test")          
    test_dir = os.path.join(data_dir, "test2d_cnn_batch")

    # Open saved meta dictionaries
    with open(os.path.join(data_path, model_dir, model_meta_file), 'r') as f: 
        model_meta = json.load(f)    
    with open(os.path.join(data_path, data_dir, "meta.json"), 'r') as f: 
        data_meta = json.load(f)    
        
    # Make the new meta dictionary
    meta = {**model_meta, 
        # ~~~~~~~~~ Paths ~~~~~~~~~#    
        "home_dir": home_dir,
        "meta_file": os.path.join(test_dir, meta_file),        
        "test_image_dir": data_meta["image_dir"],         
        "test_df_file": data_meta["target_df_file"],
        # ~~~~~~~~~ Comments ~~~~~~~~~#
        "comments": "test on validation data"
        }

    # Save dictionary to file and return some paths 
    os.makedirs(os.path.join(data_path, test_dir), exist_ok=True)
    with open(os.path.join(data_path, meta["meta_file"]), 'w') as f:
        json.dump(meta, f)

    # Absolute posix paths
    for key in [k for k in meta.keys() if "file" in k or "dir" in k]:
        meta[key] = os.path.join(data_path, meta[key])
        meta[key] = Path(PureWindowsPath(meta[key])).as_posix() # for Unix

    return meta, test_dir, model_dir


def plot_pred(pred_df, target_names, model_file, meta_file, poses_rng, **kwargs):
    plt.rcParams.update({'font.size': 18})
    n = len(target_names)
    
    fig, axes = plt.subplots(ncols=n, figsize=(7*n, 7))
    fig.suptitle(model_file.replace(os.environ['DATAPATH'],'') + '\n' + 
                 os.path.dirname(meta_file.replace(os.environ['DATAPATH'],'')))
    fig.subplots_adjust(wspace=0.3)
    n_smooth = int(pred_df.shape[0]/20)    
    for i, ax in enumerate(axes): 
        sort_df = pred_df.sort_values(by=[f"target_{i+1}"])
        ax.scatter(sort_df[f"target_{i+1}"], sort_df[f"pred_{i+1}"], s=1, c=sort_df["target_1"], cmap="inferno")
        ax.plot(sort_df[f"target_{i+1}"].rolling(n_smooth).mean(), sort_df[f"pred_{i+1}"].rolling(n_smooth).mean(), c="red")
        ax.set(xlabel=f"target {target_names[i]}", ylabel=f"predicted {target_names[i]}")
        ind = int(target_names[i][-1])-1
        ax.set_xlim(poses_rng[0][ind], poses_rng[1][ind])
        ax.set_ylim(poses_rng[0][ind], poses_rng[1][ind])
        ax.text(0.05,0.9, 'MAE='+str(sort_df[f"error_{i+1}"].mean())[0:4], transform=ax.transAxes)    
        ax.grid(True)
    return fig


def main():      
    _, test_dir, model_dir = make_meta("meta_1.json", "meta_1.json") # to get paths
   
    # Find losses and order
    trials_df = pd.read_csv(os.path.join(data_path, model_dir, "trials.csv"))
    trials_rng = trials_df.sort_values(by="loss")["trial"].tolist()
    
    # Iterate batch in order of loss
    mean_df = pd.DataFrame()
    for j, trial in enumerate(trials_rng[:]):
        meta, _, _ = make_meta(f"meta_{j+1}.json", f"meta_{trial}.json")

        # startup/load model and make predictions on test data
        cnn = CNNmodel()
        cnn.load_model(**meta)
        pred = cnn.predict_from_file(**meta)
    
        # analyze predictions
        pred_df = pd.read_csv(meta["test_df_file"])
        for i, item in enumerate(meta["target_names"], start=1):
            pred_df[f"pred_{i}"] = pred[:, i-1]
            pred_df[f"target_{i}"] = pred_df[item]
            pred_df[f"error_{i}"] = abs(pred_df[f"pred_{i}"] - pred_df[f"target_{i}"])
        pred_df.to_csv(os.path.join(data_path, test_dir, f"predictions_{j+1}.csv"))    
        fig = plot_pred(pred_df, **meta)
        fig.savefig(os.path.join(data_path, test_dir, f"errors_{j+1}.png"), bbox_inches='tight')   
        
        mean_df = mean_df.append([pred_df.mean()])
    
    # Summary plot    
    fig = plt.figure(figsize=(7*len(meta["target_names"]), 7))
    fig.suptitle(f'{model_dir}\n{test_dir}')
    for i, item in enumerate(meta["target_names"]):
        ax = fig.add_subplot(1, len(meta["target_names"]), i+1)
        ax.scatter(range(1,len(mean_df.index)+1), mean_df[f"error_{i+1}"], c='k')
        ax.set(xlabel="model", ylabel='MAE '+item)
        ax.set_ylim(bottom=0); ax.grid(True)
    fig.savefig(os.path.join(data_path, test_dir, "summary.png"), bbox_inches='tight')        


if __name__ == '__main__':
    main()
