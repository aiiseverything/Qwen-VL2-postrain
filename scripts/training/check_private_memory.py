import os, subprocess, time

def priv_mb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Private_Dirty"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0

def sample():
    ranks = workers = 0.0
    nr = nw = 0
    for pid in subprocess.run(["pgrep", "-f", "llamafactory/launcher.py"], capture_output=True, text=True).stdout.split():
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
        except Exception:
            continue
        if comm == "pt_data_worker":
            workers += priv_mb(pid); nw += 1
        elif comm == "python":
            ranks += priv_mb(pid); nr += 1
    step = subprocess.run("grep -aoE '\\| [0-9]+/4758' logs/chain_sft.log | tail -1", shell=True, capture_output=True, text=True).stdout.strip()
    anon = int(open("/sys/fs/cgroup/memory.stat").read().split("anon ")[1].split()[0]) / 1073741824
    return f"{time.strftime('%H:%M:%S')}  rank私有 {ranks/1024:.2f}G ({nr}个)  worker私有 {workers/1024:.2f}G ({nw}个)  cgroup anon {anon:.2f}G  {step}"

print("T0 ", sample())
time.sleep(240)
print("T1 ", sample())
