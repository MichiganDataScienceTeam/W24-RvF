import pathlib
import polars as pl
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from math import ceil
import numpy as np
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

rng = np.random.default_rng(42)

def get_updated_name(partition_name:str, path:str) -> str:
    return partition_name + "/" + "/".join(path.split("/")[1:])

def copy_file(partition_name, src_dir, dest_dir, path):
    src_loc = src_dir / path 
    dest_loc = dest_dir / get_updated_name(partition_name, path) 
    logger.info(f"Copying {str(src_loc)} -> {str(dest_loc)}")
    shutil.copyfile(src_loc, dest_loc)

def sample_split(df: pl.DataFrame, n: int, split: float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    sample = df.sample(n)
    num_train = int(ceil(n * split)) 

    indices = np.arange(n)
    rng.shuffle(indices)
    return sample[indices[:num_train]], sample[indices[num_train:]]

def generate_partitions(df: pl.DataFrame, n:int, split:float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    real = df.filter(pl.col("label") == 1)
    fake = df.filter(pl.col("label") == 0)

    proportion_real = len(real) / (len(real) + len(fake)) 

    num_real= int(ceil(n * proportion_real))
    num_fake = n - num_real 
    logger.info("Generating %d real samples and %d fake samples", num_real, num_fake)

    train_real, val_real = sample_split(real, num_real, split)
    train_fake, val_fake = sample_split(fake, num_fake, split)

    train = pl.concat([train_real, train_fake])
    val = pl.concat([val_real, val_fake])

    logger.info("Split: %d train / %d validation", len(train), len(val))
    return train, val

def copy_partition(partition_name, data, src_dir, dest_dir):
    partition_path = pathlib.Path(partition_name)
    partition_folder = (dest_dir / partition_name)

    if partition_folder.exists():
        logger.info(f"Removing existing files in {str(partition_folder)}")
        shutil.rmtree(partition_folder)

    partition_folder.mkdir(parents=True)
    (partition_folder / "real").mkdir()
    (partition_folder / "fake").mkdir()

    paths = data["path"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(copy_file, partition_name, src_dir, dest_dir, path) for path in paths]
        for future in as_completed(futures):
            future.result()

    # fix the filenames
    data = data.drop("path")
    data = data.with_columns(path=paths.apply(lambda path: get_updated_name(partition_name, path)))
    return data


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Dataset Preprocessing")
    parser.add_argument("source_dir", help="Path to the source directory")
    parser.add_argument("destination_dir", help="Path to the destination directory")

    args = parser.parse_args()
    
    src_dir = pathlib.Path(args.source_dir)
    dest_dir = pathlib.Path(args.destination_dir)
    
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    df = pl.read_csv(src_dir / f"train.csv", dtypes=[pl.Utf8, pl.Utf8, pl.Utf8, pl.UInt8, pl.Utf8])
    train, val = generate_partitions(df, n=10000)
    train = copy_partition("train", train, src_dir, dest_dir)
    val = copy_partition("valid", val, src_dir, dest_dir)

    train.write_csv(dest_dir / "train.csv")
    val.write_csv(dest_dir / "valid.csv")
