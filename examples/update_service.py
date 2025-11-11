import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal

import asyncio
import ray
import httpx
import torch
import torch.distributed as dist
from loguru import logger

from checkpoint_engine.ps import ParameterServer, request_inference_to_update


@contextmanager
def timer(msg: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{msg} duration: {end - start:.2f} seconds")


async def check_vllm_ready(rank: int, endpoint: str, inference_parallel_size: int, uds: str | None = None):
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return
    retry_num = 0
    print("before connection...")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(f"{endpoint}/health", timeout=600)
                response.raise_for_status()
                logger.info(f"Rank {rank}: vLLM server at {endpoint} is ready")
                break
            except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException) as e:
                retry_num += 1
                logger.warning(
                    f"Rank {rank}: fail to check vllm ready, retry {retry_num} times, error: {e}"
                )
                await asyncio.sleep(5)


def split_checkpoint_files(checkpoint_path: str, rank: int, world_size: int) -> list[str]:
    checkpoint_files = [
        os.path.join(checkpoint_path, f)
        for f in filter(lambda x: x.endswith(".safetensors"), os.listdir(checkpoint_path))
    ]
    files_per_rank = (len(checkpoint_files) + world_size - 1) // world_size
    return checkpoint_files[rank * files_per_rank : (rank + 1) * files_per_rank]


def split_tensors(checkpoint_path: str, rank: int, world_size: int) -> dict[str, torch.Tensor]:
    index_fn = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_fn) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    weights_per_rank = (len(weight_map) + world_size - 1) // world_size
    fn_tensors: dict[str, list[str]] = defaultdict(list)
    weight_keys = list(weight_map.items())
    for name, file in weight_keys[rank * weights_per_rank : (rank + 1) * weights_per_rank]:
        fn_tensors[file].append(name)
    named_tensors = {}
    #TODO(shulai): safe_open has to be imported here, otherwise object error, don't know why
    from safetensors.torch import safe_open
    for file, names in fn_tensors.items():
        with safe_open(os.path.join(checkpoint_path, file)) as f:
            for name in names:
                named_tensors[name] = f.get_tensor(name)
    return named_tensors


def req_inference(
    rank: int,
    endpoint: str,
    inference_parallel_size: int,
    uds: str | None = None,
) -> Callable[[list[tuple[str, str]]], None]:
    src = rank // inference_parallel_size * inference_parallel_size

    def req_func(socket_paths: list[tuple[str, str]]):
        if rank == src:
            request_inference_to_update(
                f"{endpoint}/collective_rpc",
                dict(socket_paths[src : src + inference_parallel_size]),
                uds=uds,
            )

    return req_func


async def update_weights(
    rank: int,
    ps: ParameterServer,
    checkpoint_name: str,
    checkpoint_files: list[str],
    named_tensors: dict[str, torch.Tensor],
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
    endpoint: str,
    save_metas_file: str | None = None,
    update_method: Literal["broadcast", "p2p", "all"] = "broadcast",
    uds: str | None = None,
):
    # await check_vllm_ready(rank, endpoint, inference_parallel_size, uds)
    dist.barrier()
    with timer("Gather metas"):
        ps.gather_metas(checkpoint_name)
    if save_metas_file and int(os.getenv("RANK")) == 0:
        with open(save_metas_file, "wb") as f:
            pickle.dump(ps.get_metas(), f)

    if update_method == "broadcast" or update_method == "all":
        with timer("Update weights without setting ranks"):
            ps.update(checkpoint_name, req_func)

    if update_method == "p2p" or update_method == "all":
        if update_method:
            # sleep 2s to wait destroy process group
            time.sleep(2)
        with timer("Update weights with setting ranks"):
            ps.update(checkpoint_name, req_func, ranks=list(range(inference_parallel_size)))

@ray.remote(num_cpus=1)
class UpdateService:
    def __init__(self, args, rank, world_size):
        self.args = args
        import os
        self.rank = rank
        self.world_size = world_size
        os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1,2,3,4"

        self.ps = ParameterServer(rank=self.rank, world_size=self.world_size, auto_pg=True)
        if os.path.exists(os.path.join(self.args.checkpoint_path, "model.safetensors.index.json")):
            with timer("Load from disk"):
                self.named_tensors = split_tensors(self.args.checkpoint_path, self.rank, self.world_size)
                self.checkpoint_files = []
        else:
            self.checkpoint_files = split_checkpoint_files(self.args.checkpoint_path, self.rank, self.world_size)
            self.named_tensors = {}
        self.ps.register_checkpoint(self.args.checkpoint_name, files=self.checkpoint_files, named_tensors=self.named_tensors)
        self.ps.init_process_group()
        self.req_func = req_inference(self.rank, self.args.endpoint, self.args.inference_parallel_size, self.args.uds)

    async def trigger_update(self):
        await update_weights(
            self.rank,
            self.ps,
            self.args.checkpoint_name,
            self.checkpoint_files,
            self.named_tensors,
            self.req_func,
            self.args.inference_parallel_size,
            self.args.endpoint,
            self.args.save_metas_file,
            self.args.update_method,
            self.args.uds,
        )

def main():
    parser = argparse.ArgumentParser(description="Pre-initialized Weight Update Service (Initialized by shulai)")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--save-metas-file", type=str, default=None)
    parser.add_argument("--load-metas-file", type=str, default=None)
    parser.add_argument("--sleep-time", type=int, default=0)
    parser.add_argument("--endpoint", type=str, default="http://localhost:19730")
    parser.add_argument("--inference-parallel-size", type=int, default=8)
    parser.add_argument("--checkpoint-name", type=str, default="my-checkpoint-iter-0")
    parser.add_argument("--update-method", type=str, default="broadcast")
    parser.add_argument("--uds", type=str, default=None)
    args = parser.parse_args()

    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "auto"
    ray.init()
    current_rank = os.getenv("RANK")
    world_size = os.getenv("WORLD_SIZE")
    actor_name = f"update_service_rank_{current_rank}"
    service = UpdateService.options(name=actor_name, namespace="weight_loader").remote(args, int(current_rank), int(world_size))
    logger.info(f"[RANK {current_rank}] Actor Created: {actor_name}")
    
    # ray.get(service.trigger_update.remote())
    while True:
        time.sleep(600)

if __name__ == "__main__":
    main()
