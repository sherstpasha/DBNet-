
python train.py --train_dirs "C:\shared\data0205\Archives020525\train_images" --train_anns "C:\shared\data0205\Archives020525\train.json" --val_dirs "C:\shared\data0205\Archives020525\test_images" --val_anns   "C:\shared\data0205\Archives020525\test.json" --batch_size 4

tensorboard --logdir runs

python infer.py --image_path "C:\Users\USER\Desktop\Figure_1_params.png" --ckpt_path "checkpoints\best_model.pth" --window_size 768 --stride 378 --prob_th 0.5 --bnd_th 0.5

python train.py --train_dirs "C:\shared\data0205\Archives020525\train_images" "C:\shared\data0205\School\train_images" --train_anns "C:\shared\data0205\Archives020525\train.json" "C:\shared\data0205\School\train.json" --val_dirs "C:\shared\data0205\Archives020525\test_images" "C:\shared\data0205\School\test_images" --val_anns  "C:\shared\data0205\Archives020525\test.json" "C:\shared\data0205\School\test.json" --batch_size 4