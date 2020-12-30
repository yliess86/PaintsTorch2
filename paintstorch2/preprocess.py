if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from paintstorch2.data.dataset.modular import ModularDataset
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
    dataset = ModularDataset.from_config(args.config, args.illustrations)
    pbar = tqdm(total=len(dataset))


    def process(i: int) -> int:
        file = f"{i:0{len(str(len(dataset)))}d}.pt"
        path = os.path.join(args.destination, file)
        torch.save(dataset[i], path)
        return 1


    workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(len(dataset)):
            pbar.update(executor.submit(process, i).result())