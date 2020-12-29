if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    from paintstorch2.data.dataset.base import Data
    from paintstorch2.data.dataset.modular import ModularPaintsTorch2Dataset
    from tqdm import tqdm

    import argparse
    import multiprocessing
    import os
    import torch
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str)
    parser.add_argument("--illustrations", type=str)
    parser.add_argument("--destination",   type=str)
    parser.add_argument("--variations",    type=int, default=25)
    args = parser.parse_args()

    if os.path.exists(args.destination):
        exit(0)
    
    os.makedirs(args.destination, exist_ok=True)

    dataset = ModularPaintsTorch2Dataset.from_config(
        args.config, args.illustrations, is_train=True,
    )

    li, lj = len(str(len(dataset))), len(str(args.variations))
    pbar = tqdm(total=len(dataset) * args.variations)


    def process(i: int, j: int) -> None:
        data = dataset[i]
        file = f"{i:0{li}d}_{j:0{lj}}.pt"
        path = os.path.join(args.destination, file)
        
        torch.save(data, path)
        pbar.update(1)


    workers = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(len(dataset)):
            for j in range(args.variations):
               executor.submit(process, i, j) 