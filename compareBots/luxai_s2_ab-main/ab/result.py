from dataclasses import dataclass
from functools import lru_cache
from typing import List

import pandas as pd

from ab.agent import AgentRevision
from ab.replay import Replay


@dataclass
class ABResult:
    rev_a: AgentRevision
    rev_b: AgentRevision
    replays: List[Replay]

    def __hash__(self) -> int:
        return hash(f"{self.rev_a.revision} {self.rev_b.revision}")

    @lru_cache()
    def get_result_df(self) -> pd.DataFrame:
        result_list = []
        win_a = 0; win_b = 0;
        lichen_a = 0; lichen_b = 0
        for r in self.replays:
            tie_flag = r.get_result() == "tie"
            end_by_player_error_flag = (
                r.get_result_by_player_error() is not None
            )
            end_by_factories_elimination_flag = (
                not end_by_player_error_flag
                and r.get_result_by_factories_elimination() is not None
            )
            end_by_running_out_of_turns = (
                not end_by_player_error_flag
                and not end_by_factories_elimination_flag
            )
            result_dict = {
                "replay_file": str(r.path),
                "player_0_hash": r.player_revisions[0],
                "player_1_hash": r.player_revisions[1],
                "seed": r.seed,
                "tie_flag": tie_flag,
                "end_by_player_error_flag": end_by_player_error_flag,
                "end_by_factories_elimination_flag": end_by_factories_elimination_flag,
                "end_by_running_out_of_turns_flag": end_by_running_out_of_turns,
            }
            for i in range(2):
                player_hash = r.player_revisions[i]
                assert player_hash in (self.rev_a.revision, self.rev_b.revision)
                player_revision = "a" if player_hash == self.rev_a.revision else "b"
                if player_revision == "a": 
                    lichen_a = lichen_a + r.get_final_n_lichen()[i]
                    print(r.get_winner_hash() == r.player_revisions[i], r.get_final_n_lichen()[i])
                    if r.get_winner_hash() == r.player_revisions[i]:
                        win_a = win_a + 1;
                if player_revision == "b": 
                    lichen_b = lichen_b + r.get_final_n_lichen()[i]
                    if r.get_winner_hash() == r.player_revisions[i]:
                        win_b = win_b + 1;                
                result_dict.update(
                    {
                        f"{player_revision}_final_n_lichen": r.get_final_n_lichen()[i],
                        f"{player_revision}_final_n_factories": r.get_final_n_factories()[i],
                        f"{player_revision}_win_flag": (
                            r.get_winner_hash() == r.player_revisions[i]
                        ),
                    }
                )
            result_list.append(result_dict)
        win_rate = 0; win_b_rate = 0 
        if (win_a + win_b) > 0 :
            win_rate = round(100.*(win_a)/(win_a + win_b),2)
            win_b_rate = round(100.*(win_b)/(win_a + win_b),2)
        lichen_rate = 0; lichen_b_rate = 0
        if (lichen_a + lichen_b) > 0 :
            lichen_rate = round(100.*(lichen_a)/(lichen_a + lichen_b),2)
            lichen_b_rate = round(100.*(lichen_b)/(lichen_a + lichen_b),2)
        print('win_a',win_a,'wr='+str(win_rate)+"%",'win_b',win_b,'lichen_a',lichen_a,'lichen_b','lr='+str(lichen_rate)+"%")
        result_dict = {
                "replay_file": "",
                "player_0_hash": r.player_revisions[0],
                "player_1_hash": r.player_revisions[1],
                "seed": "TOTAL",
                "tie_flag": "",
                "end_by_player_error_flag": "",
                "end_by_factories_elimination_flag": "",
                "end_by_running_out_of_turns_flag": "",
            }	
        result_dict.update(
                    {
                        "a_final_n_lichen": lichen_a,
                        "a_final_n_factories": "",
                        "a_win_flag": win_a,
                    }
                )			
        result_dict.update(
                    {
                        "b_final_n_lichen": lichen_b,
                        "b_final_n_factories": "",
                        "b_win_flag": win_b,
                    }
                )			
        result_list.append(result_dict)
        result_dict = {
                "replay_file": "",
                "player_0_hash": r.player_revisions[0],
                "player_1_hash": r.player_revisions[1],
                "seed": "TOTAL%",
                "tie_flag": "",
                "end_by_player_error_flag": "",
                "end_by_factories_elimination_flag": "",
                "end_by_running_out_of_turns_flag": "",
            }	
        result_dict.update(
                    {
                        "a_final_n_lichen": lichen_rate,
                        "a_final_n_factories": "",
                        "a_win_flag": win_rate,
                    }
                )			
        result_dict.update(
                    {
                        "b_final_n_lichen": lichen_b_rate,
                        "b_final_n_factories": "",
                        "b_win_flag": win_b_rate,
                    }
                )			
        result_list.append(result_dict)
        return pd.DataFrame(result_list)
