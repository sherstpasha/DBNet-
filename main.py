from train import train

if __name__ == "__main__":
    train_dirs = [r"C:\data0205\School\train_images"]
    train_anns = [r"C:\data0205\School\train.json"]
    val_dirs = [r"C:\data0205\School\test_images"]
    val_anns = [r"C:\data0205\School\test.json"]
    train(
        train_dirs,
        train_anns,
        val_dirs=val_dirs,
        val_anns=val_anns,
        epochs=50,
        bs=1,
        lr=1e-4,
        val_split=0.1,
        base_size=1024,
        img_range=256,
        crop_range=(256, 768),
        dynamic_resize=True,
    )
