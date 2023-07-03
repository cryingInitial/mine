#/bin/bash

# CIL CONFIG
NOTE="0703_ocs" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="ocs"
K_COEFF="4"
TEMPERATURE="0.125"
TRANSFORM_ON_GPU="--transform_on_gpu"
#TRANSFORM_ON_GPU=""
N_WORKER=2
FUTURE_STEPS=3
EVAL_N_WORKER=3
EVAL_BATCH_SIZE=1000
#USE_KORNIA="--use_kornia"
USE_KORNIA=""
UNFREEZE_RATE=0.25
SEEDS="1"


DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
ONLINE_ITER=3
SIGMA=0
REPEAT=1
INIT_CLS=100
USE_AMP="--use_amp"
NUM_EVAL_CLASS=10
NUM_CLASS=10

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500
    N_SMP_CLS="9" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=50 VAL_SIZE=5
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100
    BATCHSIZE=40; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    SAMPLES_PER_TASK=2500 N_TASKS=20
    # debug
    # SAMPLES_PER_TASK=40 N_TASKS=1250

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=500
    N_SMP_CLS="2" K="3" MIR_CANDS=50
    CANDIDATE_SIZE=100 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=100 
    BATCHSIZE=40; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    SAMPLES_PER_TASK=10000 N_TASKS=5


elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=100000
    N_SMP_CLS="3" K="3" MIR_CANDS=100
    CANDIDATE_SIZE=200 VAL_SIZE=2
    MODEL_NAME="resnet18" VAL_PERIOD=500 EVAL_PERIOD=200
    BATCHSIZE=32; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1
    SAMPLES_PER_TASK=20000 N_TASKS=5

elif [ "$DATASET" == "imagenet" ]; then
    MEM_SIZE=1281167
    N_SMP_CLS="3" K="3" MIR_CANDS=800
    CANDIDATE_SIZE=1000 VAL_SIZE=2
    MODEL_NAME="resnet18" EVAL_PERIOD=8000 F_PERIOD=200000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10
    SAMPLES_PER_TASK=240000 N_TASKS=5

else
    echo "Undefined setting"
    exit 1
fi



for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=0 nohup python main_new.py --mode $MODE \
    --dataset $DATASET --unfreeze_rate $UNFREEZE_RATE $USE_KORNIA --k_coeff $K_COEFF --temperature $TEMPERATURE \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS --n_tasks $N_TASKS --samples_per_task $SAMPLES_PER_TASK \
    --rnd_seed $RND_SEED --val_memory_size $VAL_SIZE --num_eval_class $NUM_EVAL_CLASS --num_class $NUM_CLASS \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE --mir_cands $MIR_CANDS \
    --memory_size $MEM_SIZE $TRANSFORM_ON_GPU --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP --n_worker $N_WORKER --future_steps $FUTURE_STEPS --eval_n_worker $EVAL_N_WORKER --eval_batch_size $EVAL_BATCH_SIZE &
done
