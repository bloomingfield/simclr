
rsync -avzh --exclude 'logs' --exclude '.git' --exclude '__pycache__' nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/simclr_tf2/cifar10_models . 
# rm staging/*

# rsync -avzh --exclude 'logs' --exclude '.git' --exclude '__pycache__' nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/simclr/tf2/runme.out . 
# # rsync -avzh nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/simclr-pytorch/*.out .
# rsync -avzh --exclude 'logs' --exclude '.git' --exclude '__pycache__' nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/simclr/tf2/runme_sup.out . 
