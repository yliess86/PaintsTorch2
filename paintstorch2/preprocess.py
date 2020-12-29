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
    args = parser.parse_args()

    if os.path.exists(args.destination):
        exit(0)
    
    os.makedirs(args.destination, exist_ok=True)

    dataset = ModularPaintsTorch2Dataset.from_config(
        args.config, args.illustrations, is_train=False,
    )
    pbar = tqdm(total=len(dataset))


    def process(i: int) -> None:
        data = dataset[i]
        file = f"{i:0{len(str(len(dataset)))}d}.pt"
        path = os.path.join(args.destination, file)
        
        torch.save(data, path)
        pbar.update(1)


    workers = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(len(dataset)):
            executor.submit(process, i)