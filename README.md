python train.py --train_dirs "C:\shared\data0205\Archives020525\train_images" "C:\shared\data0205\School\train_images" --epochs 500 --train_anns "C:\shared\data0205\Archives020525\train.json" "C:\shared\data0205\School\train.json" --val_dirs   "C:\shared\data0205\Archives020525\test_images" --val_anns   "C:\shared\data0205\Archives020525\test.json"

python train.py --train_dirs "C:\data0205\Archives020525\train_images" --train_anns  "C:\data0205\Archives020525\train.json" --val_dirs   "C:\data0205\Archives020525\test_images" --val_anns   "C:\data0205\Archives020525\test.json"

python validate.py --model_ckpt "checkpoints\best_model.pth" --test_dirs "C:\shared\data0205\Archives020525\test_images" --test_anns "C:\shared\data0205\Archives020525\test.json" --db_k 50.0 --window_size 512 --stride 256 --iou_thresh 0.5 --base_size 2048 --det_thresh_start 0.002 --det_thresh_stop 0.002 --min_area_start 10 --min_area_stop 10 --subset_frac 0.1 --device cuda --output_csv "validation_results.csv"

tensorboard --logdir runs