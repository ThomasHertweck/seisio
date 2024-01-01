# seisio

I/O operations for seismic data files in SEG-Y, SU and SEG2 format.

## Description

The seisio module provides methods to read and write seismic data in typical 
standard formats such as SEG-Y, SEG2 (read-only) or SU and can be easily extended.

The module was designed with simplicity and usability in mind. The code is 
pure Python and kept deliberately simple to get students participating our 
Geophysics classes and exercises going with Python and seismic data. The code
is not meant to offer all functionality most likely required in a commercial 
processing environment. Although best performance, highest throughput 
and minimizing memory footprint are not at the heart of this module, we have
tried to keep these topics in mind and use, for instance, memory-mapped I/O
where possible. The module has been used successfully to analyze and read 
SEG-Y data sets of approx. 10 TB in size.

## Why another seismic I/O package?

There are quite a few great Python packages available to read and/or write 
seismic data, in particular when given as SEG-Y files. Many of them are, however, 
from our perspective inherently designed to primarily deal with 3D poststack 
data leading toward seismic interpretation. Some assume a certain 3D 
inline/crossline geometry, others can only read certain pre-sorted data sets, 
or the reading of SEG-Y data seems to have been added later but was never the 
primary goal in the first place and therefore compromises were made. The seisio 
module at hand tries to avoid making any assumptions about the geometry and 
allows a user to read 2D and 3D pre- and poststack data in various flexible ways. 

## Key features

* Reads and writes SEG-Y data (with at least partial support for SEG-Y rev. 2.0,
i.e., it can handle more than 65535 samples per trace or sampling intervals 
smaller than 1 microsecond, extended textual header records or trailer records)
as IBM floats, IEEE floats, or similar.
* Reads and writes data in Seismic Unix (SU) format, both little or big endian 
(SUXDR).
* Reads SEG2 data, including non-standard strings in the descriptor blocks.
* Data are only read into memory on demand (lazy loading), not at the point of
creating the reader object; also, the file (input or output) is not kept open 
all the time, i.e., seisio itself does not need a context manager. Files should
always be in a consistent state.
* Flexible and customizable header definitions via JSON parameter file. You 
need to pick up a "float" value at byte 32 in the trace header? Or you would
like to name the SU header `cmp` instead of `cdp`? Or you have values of 
non-standard type "double" in the trace headers? No problem! You can also remap
headers when outputting files and the current trace header table does not
match the output trace header table.
* Automatic detection of endian byte order. I/O of both little- and big-endian
byte order possible.
* Automatic detection of the SEG-Y textual header encoding (ASCII or EBCDIC).
* For SEG-Y and SU data, flexible reading of traces in arbitrary order; this 
includes reading of traces based on user-defined ensembles according to trace
header mnemonics. You can, for instance, easily read CMP gathers sorted by
offset (ascending or descending), even if the traces forming the ensembles 
aren't directly located next to each other on disk.
* For reading ensembles, arbitrary filter functions can be applied. For 
instance, you can easily exclude dead traces (trace header mnemonic "trid" 
equals 2) or read only data with offsets in the range 1000 to 3000 meters.

Note: As it stands, SEG-Y or SU data need to have a constant trace length. 
The SEG-Y standard allows for the number of samples to vary trace by trace - 
this makes reading seismic data from disk rather inefficient, though. The 
module could easily be changed to work with varying trace lenghts if necessary, 
we would simply have to scan the whole file first sequentially to store the 
number of samples per trace and the byte offset within the file at which each 
trace starts. Such an approach would be similar to reading SEG2 data where trace 
pointers are stored explicitly.

## Getting Started

### Dependencies

Required: numpy, pandas 

Highly recommended: ibm2ieee, tabulate, numba

### Installation

*Install from PyPI:*

```
$> pip install seisio
```
If you would like to install also the optional dependencies (highly recommended):

```
$> pip install seisio[opt]
```

*Install directly from gitlab:*

```
$> pip install git+https://gitlab.kit.edu/thomas.hertweck/seisio.git
```

*Editable install from source:*

This version is intended for experts who would like to test the latest 
version or make modifications. Normal users should prefer to install a 
stable version.

```
$> git clone https://gitlab.kit.edu/thomas.hertweck/seisio.git
```

Once you acquired the source, you can install an editable version of seisio 
with:

```
$> cd seisio
$> pip install -e .
```

## Brief tutorial

For a demonstration of various features and much more, please visit the 
"examples" folder where several Jupyter notebooks (tutorials) are available.

Reading a (small) SEG-Y or SU file from disk into Python can be as simple as

```
import seisio

sio = seisio.input("testdata.su")
dataset = sio.read_dataset()
```
That's it, you're done. The variable `dataset` is a Numpy structured array 
that contains all the trace headers and the data themselves (don't try this
with a large data set unless you have plenty of RAM available - large data sets
should be read in a different way, see below). The code will figure out the 
type of seismic file from the suffix of the file name - if your file comes 
with an unusual suffix or no suffix at all, you may have to specify the file 
type explicitly (e.g., `filetype="SGY"`).

Extracting, for instance, the offset values for all traces is as simple as

```
offsets = dataset["offset"]
```
which will give you a 1D array of size `ntraces` that contains the offset 
values for all traces. The data themselves can be accessed by

```
data = dataset["data"]
```
as 2D Numpy array with a shape of `(ntraces, nsamples)`. Various data-related
parameters can be obtained as soon as the seisio object is established, for
instance:

```
ntraces = sio.nt
nsamples = sio.ns
sampling_interval = sio.vsi
```

If you would like to sort your data set in a certain way, this can be 
achieved by

```
dataset_sorted = np.sort(dataset, order=["offset"])
```

provided the Numpy module is imported as `np`.

Creating a file is also quite simple. If you would like to write data in 
big-endian byte order after (re-)calculating the offset header value from
the source and receiver group x-coordinates (assuming here that we deal
with a 2D seismic line and can ignore the y-components) simply requires:

```
dataset["offset"] = dataset["sx"] - dataset["gx"]
out = seisio.output("testdata_copy.su", endian=">")
out.write_traces(traces=dataset)
```
That's it. You have just created a copy of the original data in big-endian byte
order with a modified offset trace header.

Obviously, writing a SEG-Y file requires a few more steps as there are global
textual and binary file headers, possibly additional header records or trailer
records. In this case, you could use

```
out = seisio.output("testdata_copy.sgy", ns=nsamples, vsi=sampling_interval, 
                    endian=">", format=5, txtenc="ebcdic")
out.init()
out.write_traces(traces=dataset)
out.finalize()
```
This would create a SEG-Y rev. 1.0 file (default if no revision is explicitly
requested) using IEEE floats (format 5) in big-endian byte order, and the 
textual header would be encoded as EBCDIC. The `init()` method would create a 
default textual and binary file header for you (similar to SU's `segyhdrs`
command), but you could of course also get a template, 
create your own file headers, or clone file headers from another file and then 
pass them to the `init()` method, together with any extended textual header 
records (if applicable). The `finalize()` method would write any trailer records 
(if applicable; to be user-supplied as arguments); as last step, it would 
re-write the SEG-Y binary file header to reflect the correct number of traces or 
trailers in the file.

Trace headers would automatically be transferred from the SU trace header table
(input) to the SEG-Y trace header table (output). This is relatively 
straightforward as the majority of mnemonics are identical, but SU-specific 
trace headers like `d1` or `f2` would be dropped. If they need to be preserved, 
a custom-made SEG-Y trace header definition JSON file would have to be provided
that contains these header mnemonics (so they can be matched), or these header
mnemonics would have to be remapped using the `remap={"from": "to"}` parameter 
(dictionary) of the `write_traces()` method.

Theoretically, the `init()` and `finalize()` methods could be made obsolete by
forcing the user to provide all required file headers, extended file headers
and/or trailer records when creating the output object. This has deliberately
been avoided as it allows users to get header templates via 

```
textual_template = out.txthead_template
binary_template = out.binhead_template
```
that are already pre-filled with required information (such as the data format,
the number of samples, the sampling interval, the SEG-Y revision number, the
fixed-trace-length flag, header stanzas, and so on). It is perhaps a matter of
personal preference but the current choice seems somewhat more user-friendly 
and more robust in terms of setting all values required by the SEG-Y standard
correctly.

One key feature of seisio is the ability to read data in arbitrary order. In
order to achieve this, we need to scan all trace headers and create a lookup
index. If you would like to read prestack data grouped by the `xline` and
`iline` trace headers and each ensemble should be sorted by `offset`, but
you would also like the offset range to be restricted to a maximum of 4000 m, 
then this could be achieved as follows:

```
sio.create_index(group_by=["xline", "iline"], sort_by="offset",
                 filt=(lambda x: x["offset"] <= 4000))
for ens in sio.ensembles():
	... # loop through all ensembles
```
This would loop through all indiviual ensembles one at the time, and each
ensemble would have traces with the same `xline`-`iline` combination sorted
by increasing (which is the default) offset, but no offset value in any of 
the ensembles would be greater than 4000 m. Obviously, for large data sets, 
holding the lookup index in memory, although restricted to the minimum number 
of traceheader mnemonics required, possibly requires quite some memory, i.e., 
there is some overhead. This is where seismic data stored as HDF5 (or NETCDF4) 
files comes in (another of our Python modules) where trace headers can be readily 
accessed and analyzed without loading them into memory by Python modules like 
"dask" or "vaex".

Other functions that allow reading of large data files include

```
sio.read_traces(0, 1, 42, 99)
```

where you can simply specify a list of trace numbers (zero-based) to read, or

```
sio.read_batch_of_traces(start=0, ntraces=100)
```
which allows you to read a certain number of consecutive traces starting at a 
specific trace number within the file, or

```
sio.read_multibatch_of_traces(start=0, count=3, stride=4, block=2)
```
which allows you to get multiple batches of traces from the seismic file (in
this case, we would read 3 blocks of 2 traces, the first block would start at
trace number 0, and the first trace in each block would be 4 traces from the
first trace in the previous block, i.e., we would read trace numbers 0, 1, 4,
5, 8, and 9). A very simple way of looping through a file is as follows:

```
for batch in sio.batches(batch_size=1000):
	... # loop through data set in batches of 1000 traces
```
This would simply get you gathers of 1000 traces at the time, apart from 
perhaps the last gather which - dependent on the total number of traces in the
file - could be smaller.

SEG2 data sets are often relatively small, or there are individual SEG2 files 
for the survey's shots. SEG2 strings in the descriptor blocks are often (at 
least in practical terms) not complying with the SEG2 standard (many companies
add their own strings), i.e., reading of SEG2 data files into Numpy structured 
arrays with strict types or parsing SEG2 strings to put values (of a certain 
type) in a SEG-Y-like trace header table is complicated or sometimes not even 
possible, resulting in errors or loss of information. Therefore, when reading 
SEG2 data files, the seisio module returns the traces as standard 2D Numpy 
array with a separate Pandas dataframe with strings and values contained in the 
trace descriptor blocks.

The trace lengths can vary, the module will scan for the maximum number of
samples per trace and allocate a Numpy array accordingly, padding shorter 
traces with zeros where necessary. The actual number of samples per traces
is stored as additional string in the Pandas dataframe. Example:

```
sio = seisio.input("testdata.seg2")
fheader = sio.fheader   # strings of the file descriptor block
data, theaders = sio.read_all_traces()
```

## Other packages dealing with I/O of seismic data

The following list (by no means complete!) shows a few other packages dealing 
with I/O of seismic data:

* [segyio](https://github.com/equinor/segyio)
* [segysak](https://github.com/trhallam/segysak) - based on segyio for the actual I/O
* [obspy](https://github.com/obspy/obspy)
* [segpy](https://github.com/sixty-north/segpy)
* [cigsegy](https://github.com/JintaoLee-Roger/cigsegy)
* [segypy](https://github.com/cultpenguin/segypy)
* [segy-lite](https://github.com/adclose/segy_lite)
* [seg2_files](https://github.com/natstoik/seg2_files)

## Main author

Dr. Thomas Hertweck, geophysics@email.de

## License

This project is licensed under the LGPL v3.0 License - see the LICENSE.md 
file for details

