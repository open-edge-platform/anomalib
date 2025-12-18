import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
