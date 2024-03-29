# -*- coding: utf-8 -*-
#
# raster_plot.py 
# copied and arranged from nest simulator module.


""" Functions for raster plotting."""

import nest
import numpy
import matplotlib.pyplot as plt
import sys

def extract_events(data, time=None, sel=None):
    """Extract all events within a given time interval.

    Both time and sel may be used at the same time such that all
    events are extracted for which both conditions are true.

    Parameters
    ----------
    data : list
        Matrix such that
        data[:,0] is a vector of all gids and
        data[:,1] a vector with the corresponding time stamps.
    time : list, optional
        List with at most two entries such that
        time=[t_max] extracts all events with t< t_max
        time=[t_min, t_max] extracts all events with t_min <= t < t_max
    sel : list, optional
        List of gids such that
        sel=[gid1, ... , gidn] extracts all events from these gids.
        All others are discarded.

    Returns
    -------
    numpy.array
        List of events as (gid, t) tuples
    """
    val = []

    if time:
        t_max = time[-1]
        if len(time) > 1:
            t_min = time[0]
        else:
            t_min = 0

    for v in data:
        t = v[1]
        gid = v[0]
        if time and (t < t_min or t >= t_max):
            continue
        if not sel or gid in sel:
            val.append(v)

    return numpy.array(val)


def from_data(data, sel=None, **kwargs):
    """Plot raster plot from data array.

    Parameters
    ----------
    data : list
        Matrix such that
        data[:,0] is a vector of all gids and
        data[:,1] a vector with the corresponding time stamps.
    sel : list, optional
        List of gids such that
        sel=[gid1, ... , gidn] extracts all events from these gids.
        All others are discarded.
    kwargs:
        Parameters passed to _make_plot
    """
    ts = data[:, 1]
    d = extract_events(data, sel=sel)
    ts1 = d[:, 1]
    gids = d[:, 0]

    return _make_plot(ts, ts1, gids, data[:, 0], **kwargs)


def from_file(fname, **kwargs):
    """Plot raster from file.

    Parameters
    ----------
    fname : str or tuple(str) or list(str)
        File name or list of file names

        If a list of files is given, the data from them is concatenated as if
        it had been stored in a single file - useful when MPI is enabled and
        data is logged separately for each MPI rank, for example.
    kwargs:
        Parameters passed to _make_plot
    """
    if isinstance(fname, str):
        fname = [fname]

    if isinstance(fname, (list, tuple)):
        try:
            global pandas
            pandas = __import__('pandas')
            from_file_pandas(fname, **kwargs)
        except ImportError:
            from_file_numpy(fname, **kwargs)
    else:
        print('fname should be one of str/list(str)/tuple(str).')


def from_file_pandas(fname, **kwargs):
    """Use pandas."""
    data = None
    for f in fname:
        dataFrame = pandas.read_csv(
            f, sep='\s+', lineterminator='\n',
            header=None, index_col=None,
            skipinitialspace=True)
        newdata = dataFrame.values

        if data is None:
            data = newdata
        else:
            data = numpy.concatenate((data, newdata))

    return from_data(data, **kwargs)


def from_file_numpy(fname, **kwargs):
    """Use numpy."""
    data = None
    for f in fname:
        newdata = numpy.loadtxt(f)

        if data is None:
            data = newdata
        else:
            data = numpy.concatenate((data, newdata))

    return from_data(data, **kwargs)


def from_device(detec, plot_lid=False,  **kwargs):
    """
    Plot raster from a spike detector.

    Parameters
    ----------
    detec : TYPE
        Description
    plot_lid : bool, optional
        Whether to convert from local IDs
    kwargs:
        Parameters passed to _make_plot

    Raises
    ------
    nest.NESTError
    """
    if not nest.GetStatus(detec)[0]["model"] == "spike_detector":
        raise nest.NESTError("Please provide a spike_detector.")

    if nest.GetStatus(detec, "to_memory")[0]:

        ts, gids = _from_memory(detec)

        if not len(ts):
            sys.stderr.write("warning: no events recorded!\n")
            return None
            # raise nest.NESTError("No events recorded!")

        if plot_lid:
            gids = [nest.GetLID([x]) for x in gids]

        if "title" not in kwargs:
            kwargs["title"] = "Raster plot from device '%i'" % detec[0]

        # if nest.GetStatus(detec)[0]["time_in_steps"]:
        #     xlabel = "Steps"
        # else:
        #     xlabel = "Time (ms)"

        return _make_plot(ts, ts, gids, gids, **kwargs)

    elif nest.GetStatus(detec, "to_file")[0]:
        fname = nest.GetStatus(detec, "filenames")[0]
        return from_file(fname, **kwargs)

    else:
        raise nest.NESTError("No data to plot. Make sure that \
            either to_memory or to_file are set.")


def _from_memory(detec):
    ev = nest.GetStatus(detec, "events")[0]
    return ev["times"], ev["senders"]


def _make_plot(ts, ts1, gids, neurons, hist=True, hist_binwidth=5.0,
               xticks = None, yticks = None, xticks_hist = None, yticks_hist = None,
               ylabel = None, gid_offset = 0, time_offset = 0,
               grayscale=False, title=None, xlabel=None, markersize = 0.5, marker = '_'):
    """Generic plotting routine.

    Constructs a raster plot along with an optional histogram (common part in
    all routines above).

    Parameters
    ----------
    ts : list
        All timestamps
    ts1 : list
        Timestamps corresponding to gids
    gids : list
        Global ids corresponding to ts1
    neurons : list
        GIDs of neurons to plot
    hist : bool, optional
        Display histogram
    hist_binwidth : float, optional
        Width of histogram bins
    grayscale : bool, optional
        Plot in grayscale
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis
    """
    plt.figure(figsize = (8,9) if hist else (8,6))

    if grayscale:
        color_marker = marker + "k"
        color_bar = "gray"
    else:
        color_marker = marker
        color_bar = "blue"

    color_edge = "black"

    if xlabel is None:
        xlabel = "time [ms]"

    ylabel = "neuron ID" if ylabel is None else ylabel

    if hist:
        ax1 = plt.axes([0.1, 0.4, 0.85, 0.5])
        plotid = plt.plot(ts1 + time_offset, gids + gid_offset, color_marker, markersize = markersize)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        
        xlim = plt.xlim()

        plt.axes([0.1, 0.1, 0.85, 0.25])
        t_bins = numpy.arange(
            numpy.amin(ts + time_offset), numpy.amax(ts + time_offset),
            float(hist_binwidth)
        )
        n, bins = _histogram(ts + time_offset, bins=t_bins)
        num_neurons = len(numpy.unique(neurons))
        heights = 1000 * n / (hist_binwidth * num_neurons)

        plt.bar(t_bins, heights, width=hist_binwidth, color=color_bar,
                  edgecolor=color_edge)
        if xticks_hist is not None:
            plt.xticks(xticks_hist)
        if yticks_hist is not None:
            plt.yticks(yticks_hist)
        elif len(heights) >= 1:
            plt.yticks([
                int(x) for x in
                numpy.linspace(0.0, int(max(heights) * 1.1) + 5, 4)
            ])
        plt.ylabel("rate [Hz]")
        plt.xlabel(xlabel)
        plt.xlim(xlim)
        plt.axes(ax1)
    else:
        plotid = plt.plot(ts1 + time_offset, gids + gid_offset, color_marker, markersize = markersize)
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if title is None:
        plt.title("Raster plot")
    else:
        plt.title(title)

    plt.draw()

    return plotid


def _histogram(a, bins=10, bin_range=None, normed=False):
    """Calculate histogram for data.

    Parameters
    ----------
    a : list
        Data to calculate histogram for
    bins : int, optional
        Number of bins
    bin_range : TYPE, optional
        Range of bins
    normed : bool, optional
        Whether distribution should be normalized

    Raises
    ------
    ValueError
    """
    from numpy import asarray, iterable, linspace, sort, concatenate

    a = asarray(a).ravel()

    if bin_range is not None:
        mn, mx = bin_range
        if mn > mx:
            raise ValueError("max must be larger than min in range parameter")

    if not iterable(bins):
        if bin_range is None:
            bin_range = (a.min(), a.max())
        mn, mx = [mi + 0.0 for mi in bin_range]
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        bins = linspace(mn, mx, bins, endpoint=False)
    else:
        if (bins[1:] - bins[:-1] < 0).any():
            raise ValueError("bins must increase monotonically")

    # best block size probably depends on processor cache size
    block = 65536
    n = sort(a[:block]).searchsorted(bins)
    for i in range(block, a.size, block):
        n += sort(a[i:i + block]).searchsorted(bins)
    n = concatenate([n, [len(a)]])
    n = n[1:] - n[:-1]

    if normed:
        db = bins[1] - bins[0]
        return 1.0 / (a.size * db) * n, bins
    else:
        return n, bins


def show():
    """
    Show figures.

    Call plt.show() to show all figures and enter the GUI main loop.
    Python will block until all figure windows are closed again.
    You should call this function only once at the end of a script.

    See also: http://matplotlib.sourceforge.net/faq/howto_faq.html#use-show
    """
    plt.show()



