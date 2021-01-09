"""Plots graphs of timings"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns





def fps_plot(df, title):
    """plots graphs of timings"""
    df = df[~df.groupname.str.contains("ffmpeg_unblocked_decoding_speed")]
    df["groupname"] = df.groupname.str.split("_benchmark", expand=True)[0]
    sns.set(font_scale=1.8)
    g = sns.catplot(data=df, kind="bar",
                    x="groupname", y="fps", palette="dark", alpha=.6, height=5, aspect=5,
                    legend=True, legend_out=True)
    plt.xlabel(title)
    plt.savefig(title + ".png")


def combined_plot(df, title):
    """plots graphs of timings"""
    df["groupname"] = df.groupname.str.split("_benchmark", expand=True)[0]
    # This item is not needed given we have an unblocked graph
    df = df[~df.groupname.str.contains("ffmpeg_unblocked_decoding_speed")].reset_index()

    # Put the colors into a list, in order of increasing FPS (the values here are pre sorted)
    # by using a global color mapping we ensure consistent colors between graph
    palette = [GLOBAL_COLOR_MAPPINGS[group] for group in df.groupname.values]

    sns.set(font_scale=2.1)

    def barplot_err(x, y, xerr=None, yerr=None, data=None, **kwargs):
        """Plot a bar graph with hand defined symmetrical error bars"""

        _data = []
        for _i in data.index:

            _data_i = pd.concat([data.loc[_i:_i]] * 3, ignore_index=True, sort=False)
            _row = data.loc[_i]
            if xerr is not None:
                _data_i[x] = [_row[x] - _row[xerr], _row[x], _row[x] + _row[xerr]]
            if yerr is not None:
                _data_i[y] = [_row[y] - _row[yerr], _row[y], _row[y] + _row[yerr]]
            _data.append(_data_i)

        _data = pd.concat(_data, ignore_index=True, sort=False)

        _ax = sns.barplot(x=x, y=y, data=_data, ci="sd", **kwargs)

        return _ax

    _, ax = plt.subplots(figsize=(35, 10))
    _ax = barplot_err(x="groupname", y="time_for_all_frames", yerr="stddev_for_all_frames",
                      capsize=.2, data=df, ax=ax, palette=palette)
    for _, row in df.iterrows():
        print(row.name, row.time_for_all_frames - row.time_for_all_frames * 0.5)
        _ax.text(row.name, row.time_for_all_frames - row.time_for_all_frames * 0.5,
                 f"{int(round(row.fps, 0))} FPS", color="black", ha="center", va="bottom")

    plt.xlabel(title)
    plt.ylabel("Time to process 1000 frames (s)")
    plt.tight_layout()
    plt.savefig(title + ".png")

suffix = "_video_720x480.csv"

for suffix in ["_video_480x270", "_video_720x480", "_video_1920x1080"]:
    unblocked = pd.read_csv(f"benchmark_timings_unblocked{suffix}.csv")
    io = pd.read_csv(f"benchmark_timings_iolimited{suffix}.csv")
    cpu = pd.read_csv(f"benchmark_timings_cpulimited{suffix}.csv")
    #unblocked = unblocked[~unblocked.groupname.str.contains("max_possible_fps")].reset_index()

    tmp_palette = sns.color_palette("hls", len(unblocked.groupname))
    unblocked["groupname"] = unblocked.groupname.str.split("_benchmark", expand=True)[0]
    GLOBAL_COLOR_MAPPINGS = {group: color for group, color in zip(sorted(unblocked.groupname.values),
                                                                  tmp_palette)}
    print(GLOBAL_COLOR_MAPPINGS)
    combined_plot(unblocked, f"Unblocked{suffix}")
    combined_plot(io, f"IOLimited{suffix}")
    combined_plot(cpu, f"CPULimited{suffix}")
