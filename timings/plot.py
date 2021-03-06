"""Plots graphs of timings"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D




def fps_plot(df, title):
    """plots graphs of timings"""
    df = df[~df.groupname.str.contains("ffmpeg_unblocked_decoding_speed")]
    df["groupname"] = df.groupname.str.split("_benchmark", expand=True)[0]
    sns.set(font_scale=2)
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

    sns.set(font_scale=3)

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

    _, ax = plt.subplots(figsize=(40, 10))
    _ax = barplot_err(x="groupname", y="time_for_all_frames", yerr="stddev_for_all_frames",
                      capsize=.2, data=df, ax=ax, palette=palette)
    _ax.set_xticklabels([])  # remove labels on each bar
    legend_markers = []
    legend_labels = []
    for _, row in df.iterrows():
        print(row.name, row.time_for_all_frames - row.time_for_all_frames * 0.5)
        _ax.text(row.name, row.time_for_all_frames - row.time_for_all_frames * 0.5,
                 f"{int(round(row.fps, 0))} FPS", color="black", ha="center", va="bottom")
        # plot legend
        rect = Line2D([], [], marker="s", markersize=30, linewidth=0, color=palette[_])
        legend_markers.append(rect)
        legend_labels.append(row.groupname)
    print("labels", legend_labels)
    _ax.legend(legend_markers, legend_labels, bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel(title)
    plt.ylabel("Time to process 1000 frames (s)")
    plt.tight_layout()
    plt.savefig("tmp_" + title + ".png")
    return _ax

def load_df(filename):
    df = pd.read_csv(filename)
    df["fps"] = df["fps"].astype("float")
    df = df.sort_values("fps")
    df = df[df.groupname != "camgears_with_queue_benchmark"].reset_index()
    return df

for suffix in ["_video_1920x1080"]:
    unblocked = load_df(f"benchmark_timings_unblocked{suffix}.csv")
    io = load_df(f"benchmark_timings_iolimited{suffix}.csv")
    cpu = load_df(f"benchmark_timings_cpulimited{suffix}.csv")
    #unblocked = unblocked[~unblocked.groupname.str.contains("max_possible_fps")].reset_index()

    tmp_palette = sns.color_palette("hls", len(unblocked.groupname))
    unblocked["groupname"] = unblocked.groupname.str.split("_benchmark", expand=True)[0]
    GLOBAL_COLOR_MAPPINGS = {group: color for group, color in zip(sorted(unblocked.groupname.values),
                                                                  tmp_palette)}
    print(GLOBAL_COLOR_MAPPINGS)
    combined_plot(unblocked, f"Unblocked{suffix}")
    combined_plot(io, f"IOLimited{suffix}")
    combined_plot(cpu, f"CPULimited{suffix}")
