

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed so results are reproducible
N_TRIALS = 10_000


def choose_reveal_door(prize_door, first_pick):
    """Host opens a goat door that the contestant did not pick."""
    options = [d for d in (1, 2, 3) if d != first_pick and d != prize_door]
    return int(rng.choice(options))


def fair_noswitch_single_round():
    """One round where the contestant keeps their first pick."""
    prize_door = int(rng.integers(1, 4))
    first_pick = int(rng.integers(1, 4))
    reveal_door = choose_reveal_door(prize_door, first_pick)

    second_pick = first_pick  # no switching
    win = second_pick == prize_door

    return {
        "prize_door": prize_door,
        "first_pick": first_pick,
        "second_pick": second_pick,
        "reveal_door": reveal_door,
        "prize_won": "car" if win else "goat",
        "win": win,
    }


def fair_switch_single_round():
    """One round where the contestant always switches."""
    prize_door = int(rng.integers(1, 4))
    first_pick = int(rng.integers(1, 4))
    reveal_door = choose_reveal_door(prize_door, first_pick)

    # switch to the only door that is neither the first pick nor revealed
    second_pick = next(d for d in (1, 2, 3) if d != first_pick and d != reveal_door)
    win = second_pick == prize_door

    return {
        "prize_door": prize_door,
        "first_pick": first_pick,
        "second_pick": second_pick,
        "reveal_door": reveal_door,
        "prize_won": "car" if win else "goat",
        "win": win,
    }


# ---------------------------------------------------------------------------
# Generalized game
# ---------------------------------------------------------------------------

def general_single_round(game_type, num_doors, num_reveal, first_pick_input):
    """
    One round of a generalized Monty Hall game (contestant always switches).

    game_type        "fair" (every door equally likely) or "biased"
                     (door 1 most likely, last door least likely)
    num_doors        total number of doors
    num_reveal       how many goat doors the host opens
    first_pick_input a door number, or "random"
    """
    doors = np.arange(1, num_doors + 1)

    # prize probabilities
    if game_type == "biased":
        probs = doors[::-1].astype(float)  # door 1 gets the highest weight
        probs /= probs.sum()
    else:
        probs = np.full(num_doors, 1 / num_doors)

    prize_door = int(rng.choice(doors, p=probs))

    # first pick
    if first_pick_input == "random":
        first_pick = int(rng.choice(doors))
    else:
        first_pick = int(first_pick_input)

    # host reveals goat doors the contestant did not pick
    eligible = [d for d in doors if d != first_pick and d != prize_door]
    revealed_doors = rng.choice(eligible, size=num_reveal, replace=False)

    # doors still closed, excluding the first pick
    remaining = [int(d) for d in doors if d != first_pick and d not in revealed_doors]

    # second pick (always switch)
    if len(remaining) == 1:
        second_pick = remaining[0]
    elif game_type == "biased":
        # switch to the remaining door with the highest prize probability
        second_pick = max(remaining, key=lambda d: probs[d - 1])
    else:
        second_pick = int(rng.choice(remaining))

    win = second_pick == prize_door

    return {
        "bias_or_fair": game_type,
        "num_doors": num_doors,
        "num_reveal": num_reveal,
        "strategy": str(first_pick_input),
        "prize_door": prize_door,
        "first_pick": first_pick,
        "reveal_door": int(revealed_doors[0]),
        "second_pick": second_pick,
        "prize_won": "car" if win else "goat",
        "win": win,
    }


def simulate(round_fn, n=N_TRIALS, *args):
    """Run a round function n times and collect the results in a DataFrame."""
    return pd.DataFrame([round_fn(*args) for _ in range(n)])


# ---------------------------------------------------------------------------
# Run the simulations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.width", 120)

    print("=== Fair game, no switch ===")
    fair_noswitch_df = simulate(fair_noswitch_single_round)
    print(fair_noswitch_df.head())
    print(f"Win rate: {fair_noswitch_df['win'].mean():.3f}\n")

    print("=== Fair game, always switch ===")
    fair_switch_df = simulate(fair_switch_single_round)
    print(fair_switch_df.head())
    print(f"Win rate: {fair_switch_df['win'].mean():.3f}\n")

    print("=== Generalized game (always switch) ===")
    tests = [
        ("biased", 4, 2, 4),
        ("biased", 4, 2, 1),
        ("biased", 10, 2, 10),
        ("fair", 4, 1, "random"),
        ("biased", 10, 2, 1),
    ]
    summary = []
    for i, (game_type, num_doors, num_reveal, first_pick) in enumerate(tests, start=1):
        df = simulate(general_single_round, N_TRIALS,
                      game_type, num_doors, num_reveal, first_pick)
        summary.append({
            "test": i, "type": game_type, "doors": num_doors,
            "revealed": num_reveal, "first_pick": first_pick,
            "win_rate": round(df["win"].mean(), 3),
        })
    print(pd.DataFrame(summary).to_string(index=False), "\n")

    print("=== Logical indexing ===")
    combined_df = pd.concat([fair_noswitch_df, fair_switch_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # no-switch rows: second pick is the same as the first pick
    recovered_noswitch = combined_df[combined_df["first_pick"] == combined_df["second_pick"]]
    # switch rows: second pick is always different from the first pick
    recovered_switch = combined_df[combined_df["first_pick"] != combined_df["second_pick"]]

    print(f"Recovered no-switch rows: {len(recovered_noswitch)} "
          f"(win rate {recovered_noswitch['win'].mean():.3f})")
    print(f"Recovered switch rows:    {len(recovered_switch)} "
          f"(win rate {recovered_switch['win'].mean():.3f})")
