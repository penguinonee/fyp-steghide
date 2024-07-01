"""This module provides method to enter various input to the model training."""
import argparse


def arguments() -> str:
    """This function returns arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cover_path",
        default="C:\\Users\\adora\\Desktop\\FYP\\test5\\dataset\\train\\cover",
    )
    parser.add_argument(
        "--stego_path",
        default="C:\\Users\\adora\\Desktop\\FYP\\test5\\dataset\\train\\stego",
    )
    parser.add_argument(
        "--valid_cover_path",
        default="C:\\Users\\adora\\Desktop\\FYP\\test5\\dataset\\valid\\cover",
    )
    parser.add_argument(
        "--valid_stego_path",
        default=(
            "C:\\Users\\adora\\Desktop\\FYP\\test5\\dataset\\valid\\stego"
        ),
    )
    parser.add_argument("--checkpoints_dir", default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--train_size", type=int, default=15)
    parser.add_argument("--val_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)

    opt = parser.parse_args()
    return opt