"""Run sharar-eval GPQA jobs on already-launched servers: each server takes its own queue, SLOTS at a time (no reassignment).
usage: runq.py SLOTS port:model:tag,tag,... [port:model:tag,...]  (tag = <prefix>-g<cap>k-s<seed>)"""
import re, subprocess, sys, time
D = "/data/ccchang/sharar_port"
slots = int(sys.argv[1])
queues = []
for spec in sys.argv[2:]:
    port, model, tags = spec.split(":")
    queues.append([port, model, tags.split(","), []])
log = open(f"{D}/logs/runq.log", "a")
def say(m): log.write(f"[{time.strftime('%FT%T')}] {m}\n"); log.flush()
while any(q[2] or q[3] for q in queues):
    for q in queues:
        port, model, todo, running = q
        for p, t in running:
            if p.poll() is not None: say(f"done {t} rc={p.returncode}")
        q[3] = running = [(p, t) for p, t in running if p.poll() is None]
        while len(running) < slots and todo:
            t = todo.pop(0)
            cap, seed = re.search(r"-g(\d+)k-s(\d+)$", t).groups()
            p = subprocess.Popen([f"{D}/scripts/eval_custom.sh", t, port, model, "gpqa", str(int(cap) * 1024), seed, "8"],
                                 stdout=open(f"{D}/logs/eval_{t}.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)  # survives the queue being stopped
            running.append((p, t)); say(f"start {t} on port {port}")
    time.sleep(30)
say("queue done")
