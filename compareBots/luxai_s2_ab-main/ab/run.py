import subprocess
import sys
from typing import Iterable, Optional

from joblib import Parallel, delayed
from plumbum import BG, LocalPath, ProcessExecutionError, local
from plumbum.commands.modifiers import Future
from tqdm.auto import tqdm

from ab.agent import AgentRevision
from ab.replay import Replay
from ab.result import ABResult
import os


def get_replay_file_name(rev_a: AgentRevision, rev_b: AgentRevision, seed: int) -> str:
    return f"{seed}_{rev_a.revision}_{rev_b.revision}.json"


def run_game(
    script_a: LocalPath, script_b: LocalPath, seed: int, out_file: LocalPath
) -> None:
    log_out_file = out_file + ".log"
    log_err_file = out_file + ".err.log"
    cmd = local["luxai-s2"][script_a][script_b]["-v", 10]["-s", seed]["-o", out_file]
    exitcode,stdout,stderr = cmd.run(retcode=None)

    # cmd = "luxai-s2 "+str(script_a)+" "+str(script_b)+" -v 0 -s "+ str(seed)+" -o "+ str(out_file)
    # print (cmd + "   ")
    # p = subprocess.Popen(cmd)

    with open(log_out_file, "w") as text_file:
        text_file.write(stdout)
    with open(log_err_file, "w") as text_file:
        text_file.write(stderr)
    print (seed, exitcode)


def run_game_and_get_replay(
    rev_a: AgentRevision, rev_b: AgentRevision, seed: int, replay_file: LocalPath
) -> Replay:
    replay_file = replay_file
    run_game(rev_a.script_path, rev_b.script_path, seed, replay_file)
    finished = True
    return Replay(
        path=replay_file,
        player_revisions=[rev_a.revision, rev_b.revision],
        seed=seed,
        finished=finished,
    )


def run_ab(
    rev_a: AgentRevision,
    rev_b: AgentRevision,
    seeds: Iterable[int],
    replays_dir: LocalPath,
    n_jobs: Optional[int] = None,
) -> ABResult:
    if n_jobs is None:
        n_jobs = 10

    assert rev_a.revision != rev_b.revision

    replays_dir.mkdir()
    jobs = []
    for seed in seeds:
        #print (seed);
        jobs.append(
            delayed(run_game_and_get_replay)(
                rev_a,
                rev_b,
                seed,
                os.path.join(replays_dir,get_replay_file_name(rev_a, rev_b, seed)),
            )
        )
        # swap rev_a and rev_b
        jobs.append(
            delayed(run_game_and_get_replay)(
                rev_b,
                rev_a,
                seed,
                os.path.join(replays_dir,get_replay_file_name(rev_b, rev_a, seed)),
            )
        )
    replays = Parallel(n_jobs=n_jobs, backend="threading")(tqdm(jobs))
    return ABResult(rev_a, rev_b, replays)
