import argparse
from pathlib import Path
from typing import Optional

from plumbum import LocalPath
from plumbum import BG, LocalPath, ProcessExecutionError, local
from ab.agent import AgentRevision
from ab.run import run_ab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-a", type=LocalPath)
    parser.add_argument("--agent-b", type=LocalPath)
    parser.add_argument("--workdir", type=LocalPath)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, required=False)
    return parser.parse_args()


def main(
    agent_a: LocalPath,
    agent_b: LocalPath,
    workdir: LocalPath,
    n_seeds: int,
    n_jobs: Optional[int],
) -> None:
    rev_a = AgentRevision(script_path=agent_a, revision="A")
    rev_b = AgentRevision(script_path=agent_b, revision="B")
    replays_dir = workdir / "replays"
    result = run_ab(rev_a, rev_b, range(n_seeds), replays_dir)
    filename = "result_" +Path(agent_a).parts[-2]+"_"+Path(agent_b).parts[-2]+"_"+str(n_seeds)+".csv"
    result.get_result_df().to_csv(workdir / filename, index=False)
    print("Saved to ", filename)

    local.cwd.chdir(replays_dir)
    cmd = local["grep"] ["-E",''"TCFAIL|Traceback|Agent likely errored out"''] ["*.log"]
    exitcode, stdout, stderr = cmd.run(retcode=None)
    print("grep=",stdout, stderr )


if __name__ == "__main__":
    args = parse_args()
    main(args.agent_a, args.agent_b, args.workdir, args.seeds, args.jobs)
