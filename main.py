from train import train

if __name__ == "__main__":
    train_dirs = [
        # r"C:\data0205\School\train_images",
        r"C:\data0205\Archives020525\train_images",
        # r"C:\data0205\BN-HTRd\train_images",
    ]
    train_anns = [
        # r"C:\data0205\School\train.json",
        r"C:\data0205\Archives020525\train.json",
        # r"C:\data0205\BN-HTRd\train.json",
    ]
    val_dirs = [r"C:\data0205\Archives020525\test_images"]
    val_anns = [r"C:\data0205\Archives020525\test.json"]
    train(
        train_dirs,
        train_anns,
        val_dirs=val_dirs,
        val_anns=val_anns,
        epochs=50,
        bs=1,
        lr=1e-4,
        base_size=1024,
        img_range=256,
        crop_range=(256, 768),
        dynamic_resize=True,
    )
