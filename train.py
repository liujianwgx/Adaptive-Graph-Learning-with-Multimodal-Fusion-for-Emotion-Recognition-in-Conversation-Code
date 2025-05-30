from comet_ml import Experiment, Optimizer

import argparse
import torch
import os
import cogmen
import numpy as np

log = cogmen.utils.get_logger()




def main(args):

    # Create an experiment with your api key
    experiment=None
    if args.log_in_comet and not args.tuning:
        experiment = Experiment(
            api_key=args.comet_api_key,
            project_name="hyperparam-mosei",
            workspace=args.comet_workspace,
        )
        experiment.add_tag(args.tag)
    cogmen.utils.set_seed(args.seed)

    if args.emotion:
        args.data = os.path.join(
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    else:
        if args.transformers:
            args.data = os.path.join(
                args.data_dir_path,
                args.dataset,
                "transformers",
                "data_" + args.dataset + ".pkl",
            )
            print(os.path.join(args.data_dir_path, args.dataset, "transformers"))
        else:
            args.data = os.path.join(
                args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
            )

    # load data
    log.debug("Loading data from '%s'." % args.data)

    data = cogmen.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = cogmen.Dataset(data["train"], args)
    devset = cogmen.Dataset(data["dev"], args)
    testset = cogmen.Dataset(data["test"], args)

    model = cogmen.COGMEN(args,experiment).to(args.device)

    if args.use_graph_generator:
        dae_layers=torch.nn.ModuleList([model.vsw])
        dae_layers_params=list(map(id,dae_layers.parameters()))
        base_params=filter(lambda p: id(p) not in dae_layers_params,model.parameters())
        opt = cogmen.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
        opt.optimizer=torch.optim.Adam([
            {"params":base_params},
            {"params":dae_layers.parameters(),"lr":args.learning_rate_dae}],
            lr=args.learning_rate,weight_decay=args.weight_decay)
        sched = opt.get_scheduler(args.scheduler,args.epochs)
        
    else:
        opt = cogmen.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
        opt.set_parameters(model.parameters(), args.optimizer)
        sched = opt.get_scheduler(args.scheduler)

    coach = cogmen.Coach(trainset, devset, testset, model, opt, sched, args,experiment)
    log.debug("Building model...")
    if args.log_in_comet and not args.tuning:
        model_file = "./model_checkpoints/" + str(experiment.get_key()) + ".pt"
    else:
        model_file = "./model_checkpoints/model.pt"
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        default="iemocap",
        choices=["iemocap", "iemocap_4", "mosei","mosei_2"],
        help="Dataset name.",
    )
    ### adding other pre-trained text models
    parser.add_argument("--transformers", action="store_true", default=False)

    """ Dataset specific info (effects)
            -> tag_size in COGMEN.py
            -> n_speaker in COGMEN.py
            -> class_weights in classifier.py
            -> label_to_idx in Coach.py """

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    # Training parameters
    parser.add_argument(
        "--from_begin", action="store_true", help="Training from begin.", default=True
    )
    parser.add_argument("--model_ckpt", type=str, help="Training from a checkpoint.")

    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")#default=32
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",#default=adam
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler.",choices=["reduceLR","cosineLR"],
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate."#default=0.0001
    )
    parser.add_argument(
        "--learning_rate_dae" ,type=float, default=0.00003 ,help="Graph generater learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate.")#default=0.5
    parser.add_argument("--drop_rate_vsw",type=float, default=0.1, help="Dropout rate of graph generater.")
    # Model parameters
    parser.add_argument(
        "--wp",
        type=int,
        default=4,#default=5
        help="Past context window size. Set wp to -1 to use all the past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=4,#default=5
        help="Future context window size. Set wp to -1 to use all the future context.",
    )
    parser.add_argument(
        "--use_graph_generator",
        type=bool,
        default=True,
        help="Use the graph generator in model"
    )
    parser.add_argument(
        "--use_probAttention",
        type=bool,
        default=True,
        help="use the probAttention in transformer"
    )
    parser.add_argument(
        "--use_multimodel_adapter",
        type=bool,
        default=False,
        help="use the multimodel adapter in transformer"
    )
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")
    parser.add_argument(
        "--hidden_size", type=int, default=300, help="Hidden size of two layer GCN."
    )#100
    parser.add_argument(
        "--rnn",
        type=str,
        default="transformer",
        choices=["lstm", "gru", "transformer"],
        help="Type of RNN cell.",
    )
    parser.add_argument(
        "--class_weight",
        action="store_true",
        default=True, 
        help="Use class weights in nll loss.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        choices=["relational", "relative", "multi"],
        help="Type of positional encoding",
    )
    parser.add_argument(
        "--trans_encoding",
        action="store_true",
        default=False,
        help="Use dynamic embedding or not",
    )

    # Modalities
    """ Modalities effects:
        -> dimentions of input vectors in dataset.py
        -> number of heads in transformer_conv in seqcontext.py"""
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        # required=True,
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="Modalities",
    )

    # emotion
    parser.add_argument(
        "--emotion", type=str, default=None, help="emotion class for mosei"
    )

    # Model Architecture changes
    parser.add_argument("--concat_gin_gout", action="store_true", default=False)
    parser.add_argument("--seqcontext_nlayer", type=int, default=16)#default=2
    parser.add_argument("--gnn_nheads", type=int, default=7)#default=1
    parser.add_argument("--num_bases", type=int, default=7)
    parser.add_argument("--use_highway", action="store_true", default=False)
    parser.add_argument("--mask_ratio",type=int, default=15)#10
    parser.add_argument("--noise_type",type=str,default="normal",choices=["mask","normal"])
    

    # others
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    parser.add_argument("--visualize",type=bool,default=False,help="show Umap")
    parser.add_argument(
        "--log_in_comet",
        action="store_true",
        default=False,
        help="Logs the experiment data to comet.ml",
    )
    parser.add_argument(
        "--comet_api_key",
        type=str,
        default="iWv8jgfqaGYSOPMNR4MmkoUtI",
        help="comet api key, required for logging experiments on comet.ml",
    )
    parser.add_argument(
        "--comet_workspace",
        type=str,
        default="zdouble82",
        help="comet comet_workspace, required for logging experiments on comet.ml",
    )

    parser.add_argument("--use_pe_in_seqcontext", action="store_true", default=False)
    parser.add_argument("--tuning", action="store_true", default=False)
    parser.add_argument("--tag", type=str, default="hyperparameters_opt")

    args = parser.parse_args()

    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 100 + 768,
            "tv": 768 + 512,
            "av": 612,
            "atv": 100 + 768 + 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 100 + 768,
            "tv": 768 + 512,
            "av": 612,
            "atv": 100 + 768 + 512,
        },
        "mosei": {
            "a": 74,
            "t": 768,
            "v": 35,
            "at": 74 + 768,
            "tv": 768 + 35,
            "av": 74 + 35,
            "atv": 74 + 768 + 35,
        },
        "mosei_2": {
            "a": 74,
            "t": 768,
            "v": 35,
            "at": 74 + 768,
            "tv": 768 + 35,
            "av": 74 + 35,
            "atv": 74 + 768 + 35,
        },
        # "mosei": {
        #     "a": 80,
        #     "t": 768,
        #     "v": 35,
        #     "at": 80 + 768,
        #     "tv": 768 + 35,
        #     "av": 80 + 35,
        #     "atv": 80 + 768 + 35,
        # },
    }

    log.debug(args)



    main(args)
