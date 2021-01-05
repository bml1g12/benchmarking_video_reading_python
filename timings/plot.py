"""Plots graphs of timings"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cpu = pd.read_csv("benchmark_timings_cpulimited.csv")
io = pd.read_csv("benchmark_timings_iolimited.csv")
unblocked = pd.read_csv("benchmark_timings_iolimited.csv")


def plot(df, title):
    """plots graphs of timings"""
    df = df[~df.groupname.str.contains("ffmpeg_unblocked_decoding_speed")]
    df["groupname"] = df.groupname.str.split("_benchmark", expand=True)[0]
    sns.set(font_scale=1.5)
    g = sns.catplot(data=df, kind="bar",
                    x="groupname", y="fps", palette="dark", alpha=.6, height=5, aspect=5,
                    legend=True, legend_out=True)
    plt.xlabel(title)
    plt.savefig(title + ".pdf")


plot(cpu, "CPU_Limited")
plot(io, "IO_Limited")
plot(unblocked, "Unblocked")
