import os, math, json, copy, pickle, numpy as np, pandas as pd
from pathlib import Path, PureWindowsPath # for Unix compatiblity
from functools import partial
from hyperopt.pyll import scope
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK, STATUS_FAIL
from pose_models_2d.lib.models.cnn_model import CNNmodel

data_path = os.path.join(os.environ["DATAPATH"], "open", "tactile-servoing-2d-dobot")


def make_meta():
    # User-defined paths
    home_dir = os.path.join("digitac", "model_edge2d")
    train_dir = os.path.join(home_dir, "train")    
    valid_dir = os.path.join(home_dir, "test")    
    model_dir =  os.path.join(train_dir, "train2d_cnn_opt")

    # Open saved meta dictionaries
    with open(os.path.join(data_path, train_dir, "meta.json"), 'r') as f: 
        train_meta = json.load(f)           
    with open(os.path.join(data_path, valid_dir, "meta.json"), 'r') as f: 
        valid_meta = json.load(f)    

    # Make the new meta dictionary
    meta = {**train_meta, 
        # ~~~~~~~~~ Paths ~~~~~~~~~#
        "meta_file": os.path.join(model_dir, "meta.json"),
        "model_file": os.path.join(model_dir, "model.h5"),
        "train_image_dir": train_meta["image_dir"],
        "valid_image_dir": valid_meta["image_dir"],
        "train_df_file": train_meta["target_df_file"],
        "valid_df_file": valid_meta["target_df_file"],
        # ~~~~~~~~~ Model parameters ~~~~~~~~~#
        "num_conv_layers": 5,                                                                   
        "num_conv_filters": 256,                                                                
        "num_dense_layers": 1,                                                                  
        "num_dense_units": 64,  
        "activation": 'elu',                                                         
        "dropout": 0.06,                                                         
        "kernel_l1": 0.0006,                                                       
        "kernel_l2": 0.001,                                                        
        "batch_size": 8, 
        "epochs": 100,
        "patience": 10,
        "lr": 1e-4,
        "decay": 1e-6,
        "target_names": ["pose_2", "pose_6"],
        # ~~~~~~~~~ Camera settings ~~~~~~~~~#
        # "size": [128, 128], # TacTip
        "size": [160, 120], # DigiTac/Digit
        # ~~~~~~~~~ Comments ~~~~~~~~~#
        "comments": "training run on dobot mg400"
        }

    os.makedirs(os.path.join(data_path, model_dir), exist_ok=True)

    return meta, model_dir


# build hyperopt objective function
def build_objective_func(meta):
    trial = 1
    
    def objective_func(args):
        nonlocal trial
        print(f"Trial: {trial}")
        for x in args: 
            print(f'{x}:{args[x]}')
        
        # build metadata
        meta_new = {**meta, **args,
                "meta_file": meta["meta_file"].replace("meta.json", f"meta_{trial}.json"),
                "model_file": meta["model_file"].replace("model.h5", f"model_{trial}.h5")}
        
        # save metadata
        with open(os.path.join(data_path, meta_new["meta_file"]), 'w') as f: 
            json.dump(meta_new, f)             

        # Absolute posix paths
        for key in [k for k in meta_new.keys() if "file" in k or "dir" in k]:
            meta_new[key] = os.path.join(data_path, meta_new[key])
            meta_new[key] = Path(PureWindowsPath(meta_new[key])).as_posix() # for Unix

        # startup CNN, build and compile model
        cnn = CNNmodel()
        cnn.build_model(**meta_new)
        try:
            history = cnn.fit_model(**meta_new)
        except:
            results = {"loss": None, 
                       "status": STATUS_FAIL, 
                       "stopped_epoch": None, 
                       "history": None}
            print("Aborted trial: Resource exhausted error\n")
        else:
            results = {"loss": np.min(history["val_loss"]), 
                       "status": STATUS_OK, 
                       "stopped_epoch": len(history["val_loss"]),
                       "history": history}
            print("Loss: {:.2}\n".format(results["loss"])) 
        results = {**results, 'num_params': cnn._model.count_params(), 'trial': trial}

        trial += 1
        return results    
    
    return objective_func
    

def make_trials_df(trials):
    trials_df = pd.DataFrame()
    for i, trial in enumerate(trials):
        trial_params = {k: v[0] if len(v) > 0 else None for k, v in trial['misc']['vals'].items()}
        trial_row = pd.DataFrame(format_params(trial_params), index=[i])
        trial_row['loss'] = trial['result']['loss']
        trial_row['tid'] = trial['tid']
        trial_row['status'] = trial['result']['status']
        trial_row['trial'] = trial['result']['trial']
        trial_row['stopped_epoch'] = trial['result']['stopped_epoch']
        trial_row['num_params'] = trial['result']['num_params']
        trials_df = pd.concat([trials_df, trial_row])
    return trials_df


def format_params(params):
    params_conv = copy.deepcopy(params)
    params_conv["activation"] = (0,1)[params["activation"]]
    return params_conv


def main():
    # hyperopt search parameters
    space = {
        "activation":  hp.choice(label="activation", options=('relu', 'elu')),
        "dropout": hp.uniform(label="dropout", low=0, high=0.5),
        "kernel_l1": hp.loguniform(label="kernel_l1", low=-4*math.log(10), high=-math.log(10)),
        "kernel_l2": hp.loguniform(label="kernel_l2", low=-4*math.log(10), high=-math.log(10))}
    max_evals = 50
    n_startup_jobs = 20

    # perform optimization
    meta, model_dir = make_meta()     
    trials = Trials()
    obj_func = build_objective_func(meta)
    opt_params = fmin(obj_func, space, max_evals=max_evals, trials=trials, 
                      algo=partial(tpe.suggest, n_startup_jobs=n_startup_jobs))   
    
    with open(os.path.join(data_path, model_dir, "trials.pickle"), 'wb') as f:
        pickle.dump(trials, f)
    print(opt_params)
        
    # save trials history
    trials_df = make_trials_df(trials)
    trials_df.to_csv(os.path.join(data_path, model_dir, "trials.csv"), index=False)


if __name__ == '__main__':
    main()
