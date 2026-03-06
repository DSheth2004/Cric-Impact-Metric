import pandas as pd

ap = pd.read_csv('cricket_impact/all_players_impact.csv')

# Current overall top 50
print("=== CURRENT OVERALL TOP 50 ===")
top50 = ap.nlargest(50, 'overall_impact')
for i, (_, r) in enumerate(top50.iterrows()):
    bi = int(r.get('bat_innings', 0))
    boi = int(r.get('bowl_innings', 0))
    runs = int(r.get('total_runs', 0))
    wkts = int(r.get('total_wickets', 0))
    pom = int(r.get('pom_awards', 0))
    print(f"{i+1:3d}. {r['player']:25s} OI={r['overall_impact']:5.1f} BI={bi:3d} BoI={boi:3d} Runs={runs:5d} Wkts={wkts:3d} POM={pom:2d}")

# Check specific players
print("\n=== SPECIFIC PLAYERS ===")
targets = ['V Kohli', 'DA Warner', 'S Dhawan', 'KA Pollard', 'AD Russell',
           'DW Steyn', 'JJ Bumrah', 'SL Malinga', 'YS Chahal', 'HH Pandya',
           'F du Plessis', 'MS Dhoni', 'AB de Villiers', 'RG Sharma',
           'SP Narine', 'RA Jadeja', 'DJ Bravo', 'SR Watson', 'CH Gayle',
           'Rashid Khan', 'B Kumar', 'SK Raina', 'KL Rahul', 'RR Pant',
           'SA Yadav', 'Harbhajan', 'A Mishra', 'PP Chawla',
           'Shakib', 'JH Kallis', 'M Morkel', 'KH Pandya']
for name in targets:
    m = ap[ap['player'].str.contains(name, case=False, na=False)]
    if len(m) > 0:
        row = m.iloc[0]
        rank = ap[ap['overall_impact'] >= row['overall_impact']].shape[0]
        bi = int(row.get('bat_innings', 0))
        boi = int(row.get('bowl_innings', 0))
        runs = int(row.get('total_runs', 0))
        wkts = int(row.get('total_wickets', 0))
        pom = int(row.get('pom_awards', 0))
        print(f"  {row['player']:25s} Rank={rank:3d} OI={row['overall_impact']:5.1f} BI={bi:3d} BoI={boi:3d} Runs={runs:5d} Wkts={wkts:3d} POM={pom:2d}")
    else:
        print(f"  {name:25s} NOT FOUND")
