
rsync -avzh --exclude 'logs' --exclude 'cifar10_models' --exclude '.git' --exclude '__pycache__' . nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/simclr_tf2
# rm staging/*

# rsync -avzh nbloomfield@spartan.hpc.unimelb.edu.au:/data/cephfs/punim0980/simclr-pytorch/*.out .