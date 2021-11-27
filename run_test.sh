CUDA_VISIBLE_DEVICES=3 \
  python train_cl.py \
  --model_name bert_spc_cl \
  --dataset cl_res2016_2X3 \
  --num_epoch 50 \
  --seed 755 \
  --lr 2e-5 \
  --is_test 1 \
  --testfname bert_spc_cl_cl_res2016_2X3_val_type_cl2X3_acc_0.9253 \
  --type cl2X3


