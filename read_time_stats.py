import json
fname = "results/Case: 0011_stats.json"
for line in open(fname):
    stats = json.loads(line)
    for episode, details in stats.items():
        episode_no = int(episode)
        rl_stats = details['rl']
        sa_stats = details['sa']
        rl_duration = details['rl_duration']
        sa_duration =  details['sa_duration']

        for rl_stat in rl_stats:
            iter, best_cost, current_cost = rl_stat['time_step'], rl_stat['best_rl_cost'], rl_stat['state_cost']
            print(iter, best_cost, current_cost)

        for sa_stat in sa_stats:
            iter, best_cost, current_cost = sa_stat['iter']+1, sa_stat['best_sa_cost'], sa_stat['state_cost']
            print(iter, best_cost, current_cost)

