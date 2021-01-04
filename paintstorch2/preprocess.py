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
    parser.add_argument("--variations",    type=int, default=10)
    args = parser.parse_args()

    if os.path.exists(args.destination):
        exit(0)

    os.makedirs(args.destination, exist_ok=True)
    dataset = ModularDataset.from_config(args.config, args.illustrations)
    pbar = tqdm(total=len(dataset) * args.variations)


    def process(i: int, j: int) -> int:
        str_i = f"{i:0{len(str(len(dataset)))}d}"
        str_j = f"{j:0{len(str(args.variations))}d}"
        path = os.path.join(args.destination, f"{str_i}_{str_j}.pt")
        torch.save(dataset[i], path)
        return 1


    workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i in range(len(dataset)):
            for j in range(args.variations):
                pbar.update(executor.submit(process, i, j).result())