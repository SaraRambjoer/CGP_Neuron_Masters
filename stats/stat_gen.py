import os
import subprocess

# Need to do it in this dumb way because otherwise seaborn throws a stupid error after generating
# many plots
def run_interface(folder, paths):
    subprocess.call("python statistics_analyzer.py " + folder + " " + paths)

basefolder = r'C:\Users\jonod\Desktop\phenotypes_experiment_2_grouping_2'
full = False


base_sub_dirs = [f.path for f in os.scandir(basefolder)]
unified_dir = [x for x in base_sub_dirs if "unified" in x][0]
base_sub_dirs.remove(unified_dir)

base_sub_statfilepaths = [[] for _ in base_sub_dirs]

for num in range(len(base_sub_dirs)):
    dir = base_sub_dirs[num]
    sub_dirs = [f.path for f in os.scandir(dir)]
    uni_dir = [x for x in sub_dirs if "unified" in x][0]
    sub_dirs.remove(uni_dir)
    for num2 in range(len(sub_dirs)):
        sub_dir = sub_dirs[num2]
        sub_sub_dirs = [f.path for f in os.scandir(sub_dir)]
        for num3 in range(len(sub_sub_dirs)):
            sub_sub_dir = sub_sub_dirs[num3]
            statpath = os.path.join(sub_sub_dir, "statistics.yml")
            if os.path.exists(statpath):
                base_sub_statfilepaths[num].append(statpath)
                if full:
                    run_interface(sub_sub_dir, statpath)
    unified_statpath = "SPLIT".join(base_sub_statfilepaths[num])
    run_interface(uni_dir, unified_statpath)

unified_path_paths = "SPLIT".join(["SPLIT".join(x) for x in base_sub_statfilepaths])
run_interface(unified_dir, unified_path_paths)

