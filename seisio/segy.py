"""I/O of seismic files in SEG-Y format."""

import json
import logging
import numpy as np
from pathlib import Path

from . import reader
from . import tools
from . import _txtheader
from . import writer
from . import __version__

log = logging.getLogger(__name__)


_SEGYHEADSIZE = 240
_SEGYBINSIZE = 400
_SEGYTXTSIZE = 3200


class Reader(reader.Reader):
    """Class to deal with input of seismic files in SEG-Y format."""

    def __init__(self, file, **kwargs):
        """
        Initialize class Reader.

        Parameters
        ----------
        file : str or pathlib.Path
            The name of the SEG-Y input file to read.
        format : int, optional (default: None)
            Data format of SEG-Y traces, see SEG-Y standard for details.
            Usually, this is determined automatically from the binary file
            header. This parameter allows to manually override the automatic
            detection. A typical value would be 1 (IBM 32-bit floats) or 5
            (IEEE 32-bit floats).
        endian : char, optional (default: None)
            Endianess of the input file, ">" for big endian, "<" for little
            endian. If endian=None (default), then the endianess of the file
            is determined automatically. This parameter can be used to force
            a specific setting.
        fixed : bool, optional (default: None)
            Flag whether the file actually has a fixed trace length with
            identical number of samples and sampling interval on all traces.
            Usually, this is determined automatically, i.e., this flag is a
            way to override the automatic detection. Note that this class
            will only handle files with fixed trace length.
        thdef : str, optional (default: None)
            The name of the SEG-Y trace header definition JSON file. Defaults
            to the standard SEG-Y trace header definition provided by the
            seisio package.
        bhdef : str, optional (default: None)
            The name of the SEG-Y binary header definition JSON file. Defaults
            to the standard SEG-Y binary header definition provided by the
            seisio package.
        txtenc : str, optional (default: None)
            Encoding of the SEG-Y textual file header. Either "ascii" or
            "ebcdic", or None. If None (default), then the encoding of the
            textual file header is determined automatically.
        thext1 : bool, optional (default: False)
            Flag whether trace header extension 1 is used or not.
        thdef1 : str, optional (default: None)
            The name of the SEG-Y trace header extension 1 definition JSON file.
            Defaults to the standard SEG-Y header extension 1 provided by the
            seisio package.
        nthuser : int, optional (default: 0)
            The number of additional 240-byte user-defined trace headers. If
            set, this number must match the number of trace header definition
            files specified as 'thdefu'. Note that trace header extension 1
            must be present if user-defined trace headers are used, according
            to SEG-Y standard.
        thdefu : str or list of str, optional (default: None)
            The name of the SEG-Y user-defined trace header definition JSON
            file(s) if user-defined trace headers are used.
        ntxtrec : int, optional (default: None)
            The number of additional 3200-byte textual header records that
            follow the SEG-Y binary header. Usually, this is determined
            automatically, i.e., this parameter is a way to override the
            automatic detection.
        ntxtrail : int, optional (default: None)
            The number of additional 3200-byte trailer records that follow
            the actual data traces. Usually, this is determined automatically,
            i.e., this parameter is a way to override the automatic detection.
        bin_dt : str, optional (default: "dt")
            The binary header mnemonic specifying the sampling interval.
        bin_ns : str, optional (default: "ns")
            The binary header mnemonic specifying the number of samples.
        bin_format : str, otional (default: "format")
            The binary header mnemonic specifying the data format.
        bin_fixed : str, otional (default: "fixed")
            The binary header mnemonic specifying the fixed trace length flag.
        bin_iconst : str, optional (default: "iconst")
            The binary header mnemonic specifying the integer constant
            16909060_10.
        bin_segymaj : str, optional (default: "segymaj")
            The binary header mnemonic specifying the SEG-Y major revision
            number.
        bin_segymin : str, optional (default: "segymin")
            The binary header mnemonic specifying the SEG-Y minor revision
            number.
        bin_ntxthead : str, optional (default: "ntxthead")
            The binary header mnemonic specifying the number of textual header
            records.
        bin_ntrailer : str, optional (default: "ntrailer")
            The binary header mnemonic specifying the number of trailer stanza
            records.
        bin_maxtrhead : str, optional (default: "maxtrhead")
            The binary header mnemonic specifying the maximum number of
            additional 240-bytes trace headers.
        bin_byteoff : str, optional (default: "byteoff")
            The binary header mnemonic specifying the byte offset to the first
            data trace.
        bin_ens : str, optional (default: "ens")
            The binary header mnemonic specifying the extended number of
            samples.
        bin_edt : str, optional (default: "edt")
            The binary header mnemonic specifying the sextended ampling
            interval.
        bin_ntfile : str, optional (default: "ntfile")
            The binary header mnemonic specifying the number of traces in the
            file.
        mnemonic_delrt : str, optional (default: "delrt")
            The SEG-Y trace header mnemonic specifying the delay recording
            time.
        """
        super().__init__(file)
        self._sgy = self._SEGY()

        self._fp.endian = kwargs.pop("endian", None)
        self._fp.datfmt = kwargs.pop("format", None)
        self._fp.fixed = kwargs.pop("fixed", None)

        self._par["thdef"] = kwargs.pop("thdef", None)
        self._par["bhdef"] = kwargs.pop("bhdef", None)
        self._par["thext1"] = kwargs.pop("thext1", False)
        self._par["thdef1"] = kwargs.pop("thdef1", None)
        self._par["thdefu"] = kwargs.pop("thdefu", None)
        self._par["txtenc"] = kwargs.pop("txtenc", None)
        self._par["nthuser"] = kwargs.pop("nthuser", 0)
        self._par["ntxtrec"] = kwargs.pop("ntxtrec", None)
        self._par["ntxtrail"] = kwargs.pop("ntxtrail", None)

        self._par["bin_dt"] = kwargs.pop("bin_dt", "dt")
        self._par["bin_ns"] = kwargs.pop("bin_ns", "ns")
        self._par["bin_format"] = kwargs.pop("bin_format", "format")
        self._par["bin_fixed"] = kwargs.pop("bin_fixed", "fixed")
        self._par["bin_iconst"] = kwargs.pop("bin_iconst", "iconst")
        self._par["bin_segymaj"] = kwargs.pop("bin_segymaj", "segymaj")
        self._par["bin_segymin"] = kwargs.pop("bin_segymin", "segymin")
        self._par["bin_ntxthead"] = kwargs.pop("bin_ntxthead", "ntxthead")
        self._par["bin_ntrailer"] = kwargs.pop("bin_ntrailer", "ntrailer")
        self._par["bin_maxtrhead"] = kwargs.pop("bin_maxtrhead", "maxtrhead")
        self._par["bin_byteoff"] = kwargs.pop("bin_byteoff", "byteoff")
        self._par["bin_ens"] = kwargs.pop("bin_ens", "ens")
        self._par["bin_edt"] = kwargs.pop("bin_edt", "edt")
        self._par["bin_ntfile"] = kwargs.pop("bin_ntfile", "ntfile")
        self._par["mnemonic_delrt"] = kwargs.pop("mnemonic_delrt", "delrt")

        if kwargs:
            for key, val in kwargs.items():
                log.warning("Unknown argument '%s' with value '%s' encountered.", key, str(val))

        self._fp.mode = "r"

        if self._fp.endian is not None and self._fp.endian not in ["<", ">"]:
            raise ValueError(f"Unknown value '{self._fp.endian}' for argument 'endian'.")

        if self._fp.fixed is not None and self._fp.fixed is False:
            raise NotImplementedError("Support of variable-length traces "
                                      "not implemented in this class.")

        if self._par["thdef"] is None:
            self._par["thdef"] = Path(__file__).parent/"json/segy_traceheader.json"

        if self._par["bhdef"] is None:
            self._par["bhdef"] = Path(__file__).parent/"json/segy_binaryheader.json"

        with open(self._par["bhdef"], "r") as io:
            self._sgy.bhdict = json.load(io)

        if self._par["thext1"] is not None and not isinstance(self._par["thext1"], bool):
            raise TypeError("Argument 'thext1' has wrong type, should be a boolean.")
        if self._par["ntxtrec"] is not None and self._par["ntxtrec"] < 0:
            raise ValueError("Value for argument 'ntxtrec' cannot be negative.")
        if self._par["ntxtrail"] is not None and self._par["ntxtrail"] < 0:
            raise ValueError("Value for argument 'ntxtrail' cannot be negative.")
        if self._par["nthuser"] is not None and self._par["nthuser"] < 0:
            raise ValueError("Value for argument 'nthuser' cannot be negative.")

        self._sgy.txthead = _txtheader.TxtHeader(encoding=self._par["txtenc"])
        with open(self._fp.file, "rb") as fio:
            self._sgy.txthead.read(fio)
            fio.seek(0, 2)
            self._fp.filesize = fio.tell()
        self._par["txtenc"] = self._sgy.txthead.encoding
        assert self._sgy.txthead.size == _SEGYTXTSIZE

        self._fp.skip = self._sgy.txthead.size

        tmp_k, tmp_f, tmp_t = tools._parse_hdef(self._sgy.bhdict, endian="<")
        tmp_dtype_le = tools._create_dtype(tmp_k, tmp_f, titles=tmp_t)
        tmp_k, tmp_f, tmp_t = tools._parse_hdef(self._sgy.bhdict, endian=">")
        tmp_dtype_be = tools._create_dtype(tmp_k, tmp_f, titles=tmp_t)

        if self._fp.endian is None:
            self._fp.endian = self._guess_endianess(tmp_dtype_le, tmp_dtype_be)
            log.info("Input file endianess looks to be '%s' (best guess).", self._fp.endian)
        else:
            log.info("Input file endianess set to '%s'.", self._fp.endian)

        self._sgy.bhdtype = tmp_dtype_le if self._fp.endian == "<" else tmp_dtype_be
        assert self._sgy.bhdtype.itemsize == _SEGYBINSIZE

        self._get_fileattr()
        self._set_segy_dtypes()
        self._segy_nt_and_delay()

    @property
    def ntxtrec(self):
        """
        Get the number of extended textual header records.

        Returns
        -------
        int
            Number of extended header records.
        """
        return len(self._sgy.txtrec)

    @property
    def ntxtrail(self):
        """
        Get the number of trailer stanza records.

        Returns
        -------
        int
            Number of trailer stanza records.
        """
        return len(self._sgy.txtrail)

    @property
    def nthuser(self):
        """
        Get the number of user-defined trace headers.

        Returns
        -------
        int
            Number of user-defined headers.
        """
        return self._par["nthuser"]

    @property
    def thext1(self):
        """
        Get flag whether trace header extension 1 is present.

        Returns
        -------
        bool
            True if trace header extension 1 is present, otherwise False.
        """
        return self._par["thext1"]

    @property
    def txthead(self):
        """
        Get the primary SEG-Y textual file header.

        Returns
        -------
        list
            SEG-Y textual header as a list of 40 strings with 80 characters.
        """
        return self._sgy.txthead.header

    @property
    def binhead(self):
        """
        Get the SEG-Y binary file header.

        Returns
        -------
        Numpy structured array
            SEG-Y binary file header.
        """
        return self._sgy.binhead

    def log_txthead(self):
        """
        Log the primary textual file header.

        Returns
        -------
        list
            The textual header.
        """
        return self._log_txthead()

    def log_bhdef(self):
        """Log the SEG-Y binary file header definition."""
        self._log_bhdef()

    def log_binhead(self, zero=False):
        """
        Log the SEG-Y binary file header.

        Parameters
        ----------
        zero : boolean, optional (default: False)
            Whether to print binary header entries that are zero or not.

        Returns
        -------
        Numpy structured array
            The SEG-Y binary header.
        """
        return self._log_binhead(zero=zero)

    @property
    def records(self):
        """
        Get the additional SEG-Y textual header records.

        Returns
        -------
        list
            List of decoded 3200-byte-long strings.
        """
        return self.get_records()

    def get_records(self):
        """
        Get the additional SEG-Y textual header records.

        Returns
        -------
        list
            List of decoded 3200-byte-long strings.
        """
        if self.ntxtrec == 0:
            return []
        else:
            return [t.get_header(decode=True) for t in self._sgy.txtrec]

    @property
    def trailers(self):
        """
        Get the SEG-Y decoded trailer records.

        Returns
        -------
        list
            List of 3200-byte-long byte strings.
        """
        return self.get_trailers()

    def get_trailers(self, decode=True):
        """
        Get the SEG-Y trailer records.

        Parameters
        ----------
        decode : boolean, optional (default: True)
            Whether to decode the trailer or not.

        Returns
        -------
        list
            List of 3200-byte-long byte strings.
        """
        if self.ntxtrail == 0:
            return []
        else:
            return [t.get_header(decode=decode) for t in self._sgy.txtrail]

    def get_txthead(self):
        """
        Get the primary SEG-Y textual file header.

        Returns
        -------
        list
            SEG-Y textual header as a list of 40 strings with 80 characters.
        """
        return self._sgy.txthead.header

    def get_binhead(self):
        """
        Get the SEG-Y binary file header.

        Returns
        -------
        Numpy structured array
            SEG-Y binary file header.
        """
        return self._sgy.binhead

    def _guess_endianess(self, bhdtype_le, bhdtype_be):
        """
        Guess endianess of a SEG-Y file.

        Parameters
        ----------
        bhdtype_le : np.dtype
            dtype for binary header, using little endian byte ordering.
        bhdtype_be : np.dtype
            dtype for binary header, using big endian byte ordering.

        Returns
        -------
        char
            Eendianess, either "<" (little), ">" (big).
        """
        from sys import byteorder

        with open(self._fp.file, "rb") as fio:
            fio.seek(self._fp.skip, 0)
            binhead_le = np.fromfile(fio, dtype=bhdtype_le, count=1)
            fio.seek(self._fp.skip, 0)
            binhead_be = np.fromfile(fio, dtype=bhdtype_be, count=1)

        # The integer constant 16909060_10 (01020304_16). This is used to
        # allow unambiguous detection of the byte ordering to expect for this
        # SEG-Y file. For example, if this field reads as 67305985_10
        # (04030201_16) then the bytes in every Binary File Header, Trace
        # Header and Trace Data field must be reversed as they are read,
        # i.e. converting the endian-ness of the fields. If it reads
        # 33620995_10 (02010403_16) then consecutive pairs of bytes need to
        # be swapped in every Binary File Header, Trace Header and Trace Data
        # field. The byte ordering of all other portions (the Extended Textual
        # Header and Data Trailer) of the SEG-Y file is not affected by this
        # field.
        iconst_le = binhead_le[self._par["bin_iconst"]]
        iconst_be = binhead_be[self._par["bin_iconst"]]
        if iconst_be == 67305985 or iconst_le == 67305985:
            # need to swap, iconst should be 16909060
            # if host is big endian, data are small endian and vice versa
            return "<" if byteorder == "big" else ">"

        # check format
        format_le = binhead_le[self._par["bin_format"]]
        format_be = binhead_be[self._par["bin_format"]]
        if (format_be >= 1) and (format_be <= 16) and (format_le > 16):
            # likely big endian
            return ">"
        if (format_le >= 1) and (format_le <= 16) and (format_be > 16):
            # likely little endian
            return "<"

        # check dt and ns
        dt_le = binhead_le[self._par["bin_dt"]]
        dt_be = binhead_be[self._par["bin_dt"]]
        ns_le = binhead_le[self._par["bin_ns"]]
        ns_be = binhead_be[self._par["bin_ns"]]
        if (dt_be <= 10000) and (ns_be <= 20000):
            # likely big endian but less sure
            return ">"
        if (dt_le <= 10000) and (ns_le <= 20000):
            # likely little endian but less sure
            return "<"

        # still unsure; go with tradition of rev 0
        return ">"

    def _get_fileattr(self):
        """Determine certain attributes by analzying binary file header."""
        with open(self._fp.file, "rb") as fio:
            fio.seek(self._fp.skip, 0)
            self._sgy.binhead = np.fromfile(fio, dtype=self._sgy.bhdtype, count=1)

        self._fp.skip += self._sgy.bhdtype.itemsize

        self._segy_revision()
        self._segy_dataformat()
        self._segy_fixedlen()
        self._segy_headerrec()
        self._segy_trailerrec()
        self._segy_addtrhead()
        self._segy_byteoffset()
        self._segy_ns_and_si()

    def _segy_revision(self):
        """
        Determine SEG-Y revision.

        Major SEG-Y Format Revision Number. This is an 8-bit unsigned value.
        Thus for SEG-Y Revision 2.0 this will be recorded as 02_16. This field
        is mandatory for all versions of SEG-Y, although a value of zero
        indicates 'traditional' SEG-Y conforming to the 1975 standard.
        Minor SEG-Y Format Revision Number. This is an 8-bit unsigned value
        with a radix point between the first and second bytes. Thus for SEG-Y
        Revision 2.0, this will be recorded as 00_16. This field is mandatory
        for all versions of SEG-Y.
        """
        self._sgy.major = int(self._sgy.binhead[self._par["bin_segymaj"]])
        self._sgy.minor = int(self._sgy.binhead[self._par["bin_segymin"]])
        if self._sgy.major == 0:
            log.info("SEG-Y revision: original SEG-Y conforming to 1975 standard.")
        else:
            log.info("SEG-Y revision (according to binary header): %d.%d",
                     self._sgy.major, self._sgy.minor)

    def _segy_dataformat(self):
        """
        Determine SEG-Y data format.

        1 = 4-byte IBM floating-point
        2 = 4-byte, two's complement integer
        3 = 2-byte, two's complement integer
        4 = 4-byte fixed-point with gain (obsolete)
        5 = 4-byte IEEE floating-point
        6 = 8-byte IEEE floating-point
        7 = 3-byte two’s complement integer
        8 = 1-byte, two's complement integer
        9 = 8-byte, two's complement integer
        10 = 4-byte, unsigned integer
        11 = 2-byte, unsigned integer
        12 = 8-byte, unsigned integer
        15 = 3-byte, unsigned integer
        16 = 1-byte, unsigned integer
        """
        dformat = int(self._sgy.binhead[self._par["bin_format"]])

        if self._fp.datfmt is None:
            self._fp.datfmt = dformat
        else:
            if dformat != self._fp.datfmt:
                log.warning("User-requested format (%d) and format stored in binary "
                            "header (%d) differ.", self._fp.datfmt, dformat)
                log.warning("Using format specified through manual user override.")
            if self._fp.datfmt not in tools._DATAFORMAT:
                log.error("Unsupported SEG-Y data format '%d' - check for possible endian "
                          "issue or wrong 'format' argument.", self._fp.datfmt)
                raise ValueError(f"SEG-Y data format '{self._fp.datfmt}' not supported.")

        log.info("Data sample format: %s.", tools._DATAFORMAT[self._fp.datfmt]["desc"])

    def _segy_fixedlen(self):
        """
        Determine SEG-Y fixed trace length flag.

        Fixed length trace flag. A value of one indicates that all traces in
        this SEG-Y file are guaranteed to have the same sample interval,
        number of trace header blocks and trace samples, as specified in
        Binary File Header bytes 3217–3218 or 3281–3288, 3517–3518, and
        3221–3222 or 3289–3292. A value of zero indicates that the length of
        the traces in the file may vary and the number of samples in bytes
        115–116 of the Standard SEG-Y Trace Header and, if present, bytes
        137–140 of SEG-Y Trace Header Extension 1 must be examined to
        determine the actual length of each trace. This field is mandatory
        for all versions of SEG-Y, although a value of zero indicates
        'traditional' SEG-Y conforming to the 1975 standard. Irrespective of
        this flag, it is strongly recommended that corect values for the
        number of samples per trace and sample interval appear in the
        appropriate trace Trace Header locations.
        """
        segy_fixed = int(self._sgy.binhead[self._par["bin_fixed"]])

        if segy_fixed == 0:
            log.warning("SEG-Y fixed-trace-length flag is not set in binary header.")
            if self._fp.fixed is not None and self._fp.fixed is True:
                log.warning("Assuming fixed-length traces as requested by user.")
            elif self._fp.fixed is None and self._sgy.major == 0:
                self._fp.fixed = True
                log.warning("As SEG-Y major revision is 0, assuming fixed-length traces.")
            elif self._fp.fixed is None and self._sgy.major > 0:
                log.error("Fixed-trace-length flag not set but SEG-Y major rev. > 0.")
                log.error("Variable trace lengths not supported. Use parameter 'fixed' "
                          "to override if traces are in fact fixed length.")
                raise NotImplementedError("Support of variable-length traces "
                                          "not implemented in this class.")
        elif segy_fixed == 1:
            self._fp.fixed = True
            log.info("SEG-Y fixed-trace-length flag is set in binary header.")
        else:
            log.warning("SEG-Y fixed-trace-length flag set to unknown value (%d).", segy_fixed)
            if self._fp.fixed is True:
                log.warning("Assuming fixed-length traces as requested by user.")

    def _segy_headerrec(self):
        """
        Determine number of SEG-Y header records.

        Number of 3200-byte, Extended Textual File Header records following
        the Binary Header. If bytes 3521–3528 are nonzero, that field
        overrides this one. A value of zero indicates there are no Extended
        Textual File Header records (i.e. this file has no Extended Textual
        File Header(s)). A value of -1 indicates that there are a variable
        number of Extended Textual File Header records and the end of the
        Extended Textual File Header is denoted by an ((SEG: EndText)) stanza
        in the final record (Section 6.2). A positive value indicates that
        there are exactly that many Extended Textual File Header records.
        Note that, although the exact number of Extended Textual File Header
        records may be a useful piece of information, it will not always be
        known at the time the Binary Header is written and it is not
        mandatory that a positive value be recorded here or in bytes
        3521–3528. It is however recommended to record the number of records
        if possible as this makes reading more effective and supports direct
        access to traces on disk files. In the event that this number exceeds
        32767, set this field to –1 and bytes 3521–3528 to 3600+3200*(number
        of Extended Textual File Header records). Add a further 128 if a
        SEG-Y Tape Label is present.
        """
        if self._par["ntxtrec"] is None:
            segy_num_headrec = int(self._sgy.binhead[self._par["bin_ntxthead"]])
        else:
            segy_num_headrec = self._par["ntxtrec"]

        if segy_num_headrec > 0:
            # fixed number of additional header records
            with open(self._fp.file, "rb") as fio:
                for i in range(segy_num_headrec):
                    fio.seek(self._fp.skip+i*self._sgy.txthead.size, 0)
                    self._sgy.txtrec.append(_txtheader.TxtHeader(encoding=None, info="SEG-Y ext. "
                                                                 f"textual record {str(i+1)}"))
                    self._sgy.txtrec[i].read(fio)
        elif segy_num_headrec == -1:
            # variable number of add. header records
            cont_reading = True
            segy_num_headrec = 0
            with open(self._fp.file, "rb") as fio:
                while cont_reading:
                    fio.seek(self._fp.skip+segy_num_headrec*self._sgy.txthead.size, 0)
                    self._sgy.txtrec.append(_txtheader.TxtHeader(encoding=None,
                                                                 info="SEG-Y ext. textual record "
                                                                 f"{str(segy_num_headrec+1)}"))
                    self._sgy.txtrec[segy_num_headrec].read(fio)
                    headrec = self._sgy.txtrec[segy_num_headrec].get_header().casefold().strip(' ')
                    # ensure we have a valid extended header record stanza
                    if headrec[0:2] == "((":
                        segy_num_headrec += 1
                        if "((SEG: EndText))".casefold().strip(' ') in headrec:
                            cont_reading = False
                    else:
                        self._sgy.txtrec.pop()
                        cont_reading = False
                        if segy_num_headrec == 0:
                            log.warning("The SEG-Y binary header indicated a variable number "
                                        "of additional header records.")
                            log.warning("However, no valid header stanzas found.")
                            log.warning("Use argument 'ntxtrec' for manual override if required.")
        elif segy_num_headrec != 0:
            log.warning("Unknown value (%s) encountered while determining header records.",
                        str(segy_num_headrec))

        self._fp.skip += self.ntxtrec*self._sgy.txthead.size

        log.info("Number of additional textual header records: %d.", self.ntxtrec)

    def _segy_trailerrec(self):
        """
        Determine number of trailer stanza records.

        Number of 3200-byte data trailer stanza records following the last
        trace (4 byte signed integer). A value of 0 indicates there are no
        trailer records. A value of -1 indicates an undefined number of
        trailer records (0 or more) following the data. It is, however,
        recommended to record the number of trailer records if possible as
        this makes reading more efficient.
        """
        if self._par["ntxtrail"] is None:
            segy_num_trailer = int(self._sgy.binhead[self._par["bin_ntrailer"]])
        else:
            segy_num_trailer = self._par["ntxtrail"]

        if segy_num_trailer > 0:
            # fixed number of additional trailer records
            with open(self._fp.file, "rb") as fio:
                for count in range(segy_num_trailer):
                    fio.seek(0, 2)
                    self._sgy.txtrail.append(_txtheader.TxtHeader(encoding="ascii", info="SEG-Y "
                                                                  f"trailer {str(count+1)}"))
                    fio.seek(-(segy_num_trailer-count)*self._sgy.txthead.size, 2)
                    self._sgy.txtrail[count].read(fio)
        elif segy_num_trailer == -1:
            # variable number of trailer stanzas
            cont_reading = True
            segy_num_trailer = 0
            txtsize = self._sgy.txthead.size
            with open(self._fp.file, "rb") as fio:
                while cont_reading:
                    fio.seek(-(txtsize+segy_num_trailer*txtsize), 2)
                    self._sgy.txtrail.append(_txtheader.TxtHeader(encoding="ascii",
                                                                  info="SEG-Y trailer "
                                                                  f"{str(segy_num_trailer+1)}"))
                    self._sgy.txtrail[segy_num_trailer].read(fio)
                    stanza = self._sgy.txtrail[segy_num_trailer].get_header().casefold().strip(' ')
                    # ensure we have a valid stanza
                    if "((SEG:User Data".casefold().strip(' ') in stanza[0:18]:
                        segy_num_trailer += 1
                    else:
                        self._sgy.txtrail.pop()
                        cont_reading = False
                        if segy_num_trailer == 0:
                            log.warning("The SEG-Y binary header indicated a variable number "
                                        "of data trailer stanza records.")
                            log.warning("However, no valid trailer stanzas '((SEG:User Data' found.")
                            log.warning("Use argument 'ntrailer' for manual override if required.")
        elif segy_num_trailer != 0:
            log.warning("Unknown value (%s) encountered while determining "
                        "trailer records.", str(segy_num_trailer))
        if segy_num_trailer > 0:
            log.warning("This SEG-Y file has trailer stanza records. If they contain binary data,")
            log.warning("decoding them is impossible without parsing the XML description. The")
            log.warning("trailer records might be returned as raw buffers for the user to decode.")

        log.info("Number of trailer stanza records: %d.", self.ntxtrail)

    def _segy_addtrhead(self):
        """
        Determine number of additional trace headers.

        Maximum number of additional 240 byte trace headers. A value of zero
        indicates there are no additional 240 byte trace headers. The actual
        number for a given trace may be supplied in bytes 157–158 of SEG-Y
        Trace Header Extension 1.
        """
        segy_num_trhead = int(self._sgy.binhead[self._par["bin_maxtrhead"]])
        if segy_num_trhead == 0:
            # no additional trace headers present
            if self._par["thext1"]:
                log.warning("Binary header indicates there are no additional 240-byte "
                            "trace headers.")
                log.warning("However, argument 'thext1' was provided, implying the "
                            "presence of trace header extension 1.")
                log.warning("User-supplied parameter overturns binary header value.")
                # self._par["thext1"] already set
            if self._par["nthuser"] > 0:
                log.warning("Binary header indicates there are no additional 240-byte "
                            "user-defined trace headers.")
                log.warning("However, argument 'nthuser' was provided, implying the "
                            "presence of %d user-defined trace headers.", self._par["nthuser"])
                log.warning("User-supplied parameter overturns binary header value.")
                # if user-defined headers are present, extension 1 must be present, too
                self._par["thext1"] = True
        elif segy_num_trhead == 1:
            # one additional trace header present, must be extension 1
            self._par["thext1"] = True
            if self._par["nthuser"] > 0:
                log.warning("Binary header indicates there is one additional 240-byte "
                            "trace header (extension 1).")
                log.warning("However, argument 'nthuser' was provided, implying the presence "
                            "of additional %d user-defined trace headers.", self._par["nthuser"])
                log.warning("User-supplied parameter overturns binary header value.")
        else:
            # binary header indicated >=2 additonal trace headers
            self._par["thext1"] = True
            self._par["nthuser"] = segy_num_trhead-1

        msg = " " if self._par["thext1"] else " not "
        log.info("SEG-Y trace header extension 1 is%spresent.", msg)
        log.info("Number of user-defined trace headers: %d", self._par["nthuser"])

    def _segy_byteoffset(self):
        """
        Determine byte offset.

        Byte offset of first trace relative to start of file or stream if
        known, otherwise zero. (64-bit unsigned integer value) This byte
        count will include the initial 3600 bytes of the Textual and this
        Binary File Header plus the Extended Textual Header if present.
        When nonzero, this field overrides the byte offset implied by any
        nonnegative number of Extended Textual Header records present in
        bytes 3505–3506.
        """
        segy_byte_offset = int(self._sgy.binhead[self._par["bin_byteoff"]])
        byte_offset = self._fp.skip
        if segy_byte_offset > 0:
            self._fp.skip = segy_byte_offset
            if segy_byte_offset != byte_offset:
                log.warning("Byte offset to first trace as stored in SEG-Y "
                            "binary header differs from calculated offset.")
                log.warning("Using value from header. Header: %d bytes, calculated: "
                            "%d bytes.", segy_byte_offset, byte_offset)

        log.info("Byte offset of first trace relative to start of file: %d bytes.",
                 self._fp.skip)

    def _segy_ns_and_si(self):
        """
        Determine number of samples and sampling interval.

        Number of samples per data trace. Note: The sample interval and number
        of samples in the Binary File Header should be for the primary set of
        seismic data traces in the file.

        Sampling interval. Microseconds (μs) for time data, Hertz (Hz) for
        frequency data, meters (m) or feet (ft) for depth data. Extended sample
        interval, IEEE double precision (64-bit). If nonzero, this overrides
        the sample interval in bytes 3217–3218 with the same units.
        """
        self._dp.ns = int(self._sgy.binhead[self._par["bin_ns"]])
        extended_ns = int(self._sgy.binhead[self._par["bin_ens"]])
        if extended_ns > 0:
            self._dp.ns = extended_ns

        log.info("Number of samples per data trace: %d.", self._dp.ns)

        self._dp.si = int(self._sgy.binhead[self._par["bin_dt"]])
        extended_dt = np.float64(self._sgy.binhead[self._par["bin_edt"]])
        if extended_dt != 0:
            self._dp.si = extended_dt

        log.info("Sampling interval: %s (unit as per SEG-Y standard).", str(self._dp.si))

    def _segy_nt_and_delay(self):
        """Determine number of traces and delay recording time."""
        segy_num_traces = int(self._sgy.binhead[self._par["bin_ntfile"]])
        non_data = self._fp.skip + self.ntxtrail * self._sgy.txthead.size
        num_traces = int((self.fsize-non_data)/self.trsize)

        if segy_num_traces > 0:
            self._dp.nt = segy_num_traces
            if segy_num_traces != num_traces:
                log.warning("Number of traces as stored in binary header differs from "
                            "calculated number of traces. Using value from header.")
                log.warning("Stored in header: %d, calculated: %d.", segy_num_traces, num_traces)
        else:
            self._dp.nt = num_traces
            if (self.fsize-non_data) % self.trsize != 0:
                log.warning("Length mismatch encountered in file %s; trying to continue.",
                            self._fp.file)
                log.warning("Filesize: %d bytes, trace size: %d bytes, headers and trailers:"
                            " %d bytes.", self.fsize, self.trsize, non_data)

        log.info("Number of data traces in file: %d.", self._dp.nt)

        with open(self._fp.file, "rb") as fio:
            fio.seek(self._fp.skip, 0)
            headers = np.fromfile(fio, dtype=self._tr.thdtype, count=1)
            self._dp.delay = headers[self._par["mnemonic_delrt"]][0]

        log.info("Delay (on first trace): %s (unit as per SEG-Y standard).", str(self._dp.delay))


class Writer(writer.Writer):
    """Class to deal with output of seismic files in SEG-Y format."""

    def __init__(self, file, **kwargs):
        """
        Initialize class Writer.

        Parameters
        ----------
        file : str or pathlib.Path
            The name of the SEG-Y output file to write.
        ns : int
            Number of samples per output trace.
        vsi : int
            (Vertical) sampling interval (typically in microunits).
        endian : char, optional (default: ">")
            Endianess of the input file, ">" for big endian, "<" for little
            endian, "=" for native endian.
        format : int, optional (default: 5)
            Data format of SEG-Y traces, see SEG-Y standard for details.
        segymaj : int, optional (default: 1)
            SEG-Y major revision number.
        segymin : int, optional (default: 0)
            SEG-Y minor revision number.
        txtenc : str, optional (default: "ascii")
            Encoding of the SEG-Y textual file header. Either "ascii" or
            "ebcdic".
        thdef : str, optional (default: None)
            The name of the SEG-Y trace header definition JSON file. Defaults
            to the standard SEG-Y trace header definition provided by the
            seisio package.
        bhdef : str, optional (default: None)
            The name of the SEG-Y binary header definition JSON file. Defaults
            to the standard SEG-Y binary header definition provided by the
            seisio package.
        thext1 : bool, optional (default: False)
            Flag whether trace header extension 1 is used or not.
        thdef1 : str, optional (default: None)
            The name of the SEG-Y trace header extension 1 definition JSON file.
            Defaults to the standard SEG-Y header extension 1 provided by the
            seisio package.
        nthuser : int, optional (default: 0)
            The number of additional 240-byte user-defined trace headers. If
            set, this number must match the number of trace header definition
            files specified as 'thdefu'. Note that trace header extension 1
            must be present if user-defined trace headers are used, according
            to SEG-Y standard.
        thdefu : str or list of str, optional (default: None)
            The name of the SEG-Y user-defined trace header definition JSON
            file(s) if user-defined trace headers are used.
        ntxtrec : int, optional (default: 0)
            The number of additional 3200-byte textual header records that
            follow the SEG-Y binary header. Usually, this is determined
            automatically, i.e., this parameter is a way to override the
            automatic detection.
        ntxtrail : int, optional (default: 0)
            The number of additional 3200-byte trailer records that follow
            the actual data traces. Usually, this is determined automatically,
            i.e., this parameter is a way to override the automatic detection.
        bin_dt : str, optional (default: "dt")
            The binary header mnemonic specifying the sampling interval.
        bin_ns : str, optional (default: "ns")
            The binary header mnemonic specifying the number of samples.
        bin_format : str, otional (default: "format")
            The binary header mnemonic specifying the data format.
        bin_fixed : str, otional (default: "fixed")
            The binary header mnemonic specifying the fixed trace length flag.
        bin_iconst : str, optional (default: "iconst")
            The binary header mnemonic specifying the integer constant
            16909060_10.
        bin_segymaj : str, optional (default: "segymaj")
            The binary header mnemonic specifying the SEG-Y major revision
            number.
        bin_segymin : str, optional (default: "segymin")
            The binary header mnemonic specifying the SEG-Y minor revision
            number.
        bin_ntxthead : str, optional (default: "ntxthead")
            The binary header mnemonic specifying the number of textual header
            records.
        bin_ntrailer : str, optional (default: "ntrailer")
            The binary header mnemonic specifying the number of trailer stanza
            records.
        bin_maxtrhead : str, optional (default: "maxtrhead")
            The binary header mnemonic specifying the maximum number of
            additional 240-bytes trace headers.
        bin_byteoff : str, optional (default: "byteoff")
            The binary header mnemonic specifying the byte offset to the first
            data trace.
        bin_ens : str, optional (default: "ens")
            The binary header mnemonic specifying the extended number of
            samples.
        bin_edt : str, optional (default: "edt")
            The binary header mnemonic specifying the sextended ampling
            interval.
        bin_ntfile : str, optional (default: "ntfile")
            The binary header mnemonic specifying the number of traces in the
            file.
        """
        mode = "w"
        super().__init__(file, mode)

        self._sgy = self._SEGY()

        self._fp.mode = mode
        self._fp.fixed = True
        self._fp.endian = kwargs.pop("endian", ">")
        self._fp.datfmt = kwargs.pop("format", 5)
        self._dp.ns = kwargs.pop("ns", None)
        self._dp.si = kwargs.pop("vsi", None)

        self._par["segymaj"] = kwargs.pop("segymaj", 1)
        self._par["segymin"] = kwargs.pop("segymin", 0)
        self._par["thdef"] = kwargs.pop("thdef", None)
        self._par["bhdef"] = kwargs.pop("bhdef", None)
        self._par["thext1"] = kwargs.pop("thext1", False)
        self._par["thdef1"] = kwargs.pop("thdef1", None)
        self._par["thdefu"] = kwargs.pop("thdefu", None)
        self._par["txtenc"] = kwargs.pop("txtenc", "ascii")
        self._par["nthuser"] = kwargs.pop("nthuser", 0)
        self._par["ntxtrec"] = kwargs.pop("ntxtrec", 0)
        self._par["ntxtrail"] = kwargs.pop("ntxtrail", 0)

        self._par["bin_dt"] = kwargs.pop("bin_dt", "dt")
        self._par["bin_ns"] = kwargs.pop("bin_ns", "ns")
        self._par["bin_format"] = kwargs.pop("bin_format", "format")
        self._par["bin_fixed"] = kwargs.pop("bin_fixed", "fixed")
        self._par["bin_iconst"] = kwargs.pop("bin_iconst", "iconst")
        self._par["bin_segymaj"] = kwargs.pop("bin_segymaj", "segymaj")
        self._par["bin_segymin"] = kwargs.pop("bin_segymin", "segymin")
        self._par["bin_ntxthead"] = kwargs.pop("bin_ntxthead", "ntxthead")
        self._par["bin_ntrailer"] = kwargs.pop("bin_ntrailer", "ntrailer")
        self._par["bin_maxtrhead"] = kwargs.pop("bin_maxtrhead", "maxtrhead")
        self._par["bin_byteoff"] = kwargs.pop("bin_byteoff", "byteoff")
        self._par["bin_ens"] = kwargs.pop("bin_ens", "ens")
        self._par["bin_edt"] = kwargs.pop("bin_edt", "edt")
        self._par["bin_ntfile"] = kwargs.pop("bin_ntfile", "ntfile")

        if kwargs:
            for key, val in kwargs.items():
                log.warning("Unknown argument '%s' with value '%s' encountered.", key, str(val))

        self._par_check()
        self._set_segy_dtypes()

        log.info("Output file endianess set to '%s'.", self._fp.endian)
        log.info("Number of additional textual header records: %d.", self.ntxtrec)
        log.info("Byte offset of first trace relative to start of file: %d bytes.",
                 self._fp.skip)
        log.info("Number of trailer stanza records: %d.", self.ntxtrail)
        msg = " " if self._par["thext1"] else " not "
        log.info("SEG-Y trace header extension 1 is%spresent.", msg)
        log.info("Number of user-defined trace headers: %d", self._par["nthuser"])
        log.info("Creating file according to SEG-Y rev. %d.%d.",
                 self._par["segymaj"], self._par["segymin"])

    @property
    def ntxtrec(self):
        """
        Get the number of extended textual header records.

        Returns
        -------
        int
            Number of extended header records.
        """
        return len(self._sgy.txtrec)

    @property
    def ntxtrail(self):
        """
        Get the number of trailer stanza records.

        Returns
        -------
        int
            Number of trailer stanza records.
        """
        return len(self._sgy.txtrail)

    @property
    def nthuser(self):
        """
        Get the number of user-defined trace headers.

        Returns
        -------
        int
            Number of user-defined headers.
        """
        return self._par["nthuser"]

    @property
    def thext1(self):
        """
        Get flag whether trace header extension 1 is present.

        Returns
        -------
        bool
            True if trace header extension 1 is present, otherwise False.
        """
        return self._par["thext1"]

    def _par_check(self):
        """Check parameters."""
        if self._fp.endian not in ["<", ">", "="]:
            raise ValueError(f"Unknown value '{self._fp.endian}' for argument 'endian'.")

        if self._fp.endian == "=":
            self._fp.endian = tools._native_endian()

        if self._fp.datfmt not in tools._DATAFORMAT:
            raise ValueError(f"Unknown or unsupported 'format' value {self._fp.datfmt}.")

        if self._dp.ns is None or self._dp.ns <= 0:
            raise ValueError("Need parameter 'ns' greater than zero.")

        if self._dp.si is None or self._dp.si <= 0:
            raise ValueError("Need parameter 'vsi' greater than zero.")

        log.info("Output number of samples per data trace: %d.", self.ns)
        log.info("Output data sample format: %s.", tools._DATAFORMAT[self._fp.datfmt]["desc"])
        if self._fp.datfmt == 1:
            log.warning("IBM floats (format 1) are deprecated. "
                        "Consider using IEEE floats (format 5).")

        if self._fp.endian == "<" and self._fp.datfmt == 1:
            log.warning("Resetting endian to '>' for format %s.",
                        tools._DATAFORMAT[self._fp.datfmt]["desc"])
            self._fp.endian = ">"

        if self._par["segymaj"] not in [0, 1, 2]:
            raise ValueError("Parameter 'segymaj' needs to be 0, 1 or 2.")
        if self._par["segymin"] not in [0, 1]:
            raise ValueError("Parameter 'segymin' needs to be 0 or 1.")

        self._sgy.txthead = _txtheader.TxtHeader(encoding=self._par["txtenc"])
        self._fp.skip = self._sgy.txthead.size

        if self._par["bhdef"] is None:
            self._par["bhdef"] = Path(__file__).parent/"json/segy_binaryheader.json"

        with open(self._par["bhdef"], "r") as io:
            self._sgy.bhdict = json.load(io)

        k, f, t = tools._parse_hdef(self._sgy.bhdict, endian=self._fp.endian)
        self._sgy.bhdtype = tools._create_dtype(k, f, titles=t)
        assert self._sgy.bhdtype.itemsize == _SEGYBINSIZE

        self._fp.skip += self._sgy.bhdtype.itemsize

        if self._par["thdef"] is None:
            self._par["thdef"] = Path(__file__).parent/"json/segy_traceheader.json"

        if self._par["ntxtrec"] is not None and self._par["ntxtrec"] < 0:
            raise ValueError("Value for argument 'ntxtrec' cannot be negative.")

        for i in np.arange(self._par["ntxtrec"]):
            self._sgy.txtrec.append(_txtheader.TxtHeader(encoding=self._par["txtenc"],
                                                         info="SEG-Y ext. textual "
                                                         f"record {str(i+1)}"))
            self._fp.skip += self._sgy.txtrec[i].size

        if self._par["thext1"] is not None and not isinstance(self._par["thext1"], bool):
            raise TypeError("Argument 'thext1' has wrong type, should be a boolean.")

        if self._par["ntxtrail"] is not None and self._par["ntxtrail"] < 0:
            raise ValueError("Value for argument 'ntxtrail' cannot be negative.")

        for i in np.arange(self._par["ntxtrail"]):
            self._sgy.txtrail.append(_txtheader.TxtHeader(encoding="ascii", info="SEG-Y "
                                                          f"trailer {str(i+1)}"))

        if self._par["nthuser"] is not None and self._par["nthuser"] < 0:
            raise ValueError("Value for argument 'nthuser' cannot be negative.")

        if self._par["nthuser"] > 0 and self._par["thext1"] is False:
            log.warning("User-defined trace headers require header extension 1 to be present.")
            log.warning("Setting parameter 'thext1' to 'True'.")
            self._par["thext1"] = True

        if self._fp.endian == "<" and self._par["segymaj"] in [0, 1]:
            log.warning("Little-endian byte ordering not strictly compliant with SEG-Y "
                        "rev. %d.", self._par["segymaj"])

        if self.ntxtrec > 0 and self._par["segymaj"] == 0:
            log.warning("Additional textual file header records not compliant with SEG-Y "
                        "rev. 0.")
            log.warning("Setting parameter 'segymaj' to '1'.")
            self._par["segymaj"] = 1

        if (self._par["nthuser"] > 0 or self._par["thext1"]) and self._par["segymaj"] in [0, 1]:
            log.warning("Additional trace headers not compliant with SEG-Y rev. 0 or 1.")
            log.warning("Setting parameter 'segymaj' to '2'.")
            self._par["segymaj"] = 2

        if self._par["ntxtrail"] > 0 and self._par["segymaj"] in [0, 1]:
            log.warning("Trailer records not compliant with SEG-Y rev. 0 or 1.")
            log.warning("Setting parameter 'segymaj' to '2'.")
            self._par["segymaj"] = 2

        if self._dp.ns > 65535 and self._par["segymaj"] in [0, 1]:
            log.warning("Number of samples (%d) not compliant with SEG-Y rev. 0 or 1.",
                        self._dp.ns)
            log.warning("Setting parameter 'segymaj' to '2'.")
            self._par["segymaj"] = 2

    @property
    def txthead_template(self):
        """
        Provide a template for a primary SEG-Y textual file header.

        Returns
        -------
        list
            A list of strings, 40 card images (strings), each of 80 characters.
        """
        from . import segy_txthead_template
        return segy_txthead_template(major_version=self._par["segymaj"],
                                     minor_version=self._par["segymin"], fill=True)

    @property
    def txtrec_template(self):
        """
        Provide a template for an additional SEG-Y textual file header.

        Note: The difference to txthead_template(), which provides a primary
        textual file header, is the lack of the 'Cxx' convention at the
        beginning of each card image.

        Returns
        -------
        list
            A list of strings, 40 card images (strings), each of 80 characters.
        """
        from . import segy_txthead_template
        return segy_txthead_template(major_version=self._par["segymaj"],
                                     minor_version=self._par["segymin"], fill=False)

    def log_bhdef(self):
        """Log the SEG-Y binary file header definition."""
        self._log_bhdef()

    @property
    def binhead_template(self):
        """
        Provide a template for a SEG-Y binary file header.

        Returns
        -------
        Numpy structured array
            SEG-Y binary file header.
        """
        self._sgy.binhead = np.zeros((1, ), dtype=self._sgy.bhdtype)

        self._sgy.binhead[self._par["bin_format"]] = self._fp.datfmt
        if self._dp.si.is_integer():
            self._sgy.binhead[self._par["bin_dt"]] = self._dp.si
            self._sgy.binhead[self._par["bin_edt"]] = 0
        else:
            self._sgy.binhead[self._par["bin_dt"]] = 0
            self._sgy.binhead[self._par["bin_edt"]] = self._dp.si

        if self._dp.ns > 65535:
            self._sgy.binhead[self._par["bin_ns"]] = 0
            self._sgy.binhead[self._par["bin_ens"]] = self._dp.ns
        else:
            self._sgy.binhead[self._par["bin_ns"]] = self._dp.ns
            self._sgy.binhead[self._par["bin_ens"]] = 0

        self._sgy.binhead[self._par["bin_segymaj"]] = self._par["segymaj"]
        self._sgy.binhead[self._par["bin_segymin"]] = self._par["segymin"]
        self._sgy.binhead[self._par["bin_fixed"]] = 1
        self._sgy.binhead[self._par["bin_ntxthead"]] = self.ntxtrec
        self._sgy.binhead[self._par["bin_maxtrhead"]] = 0
        if self.thext1:
            self._sgy.binhead[self._par["bin_maxtrhead"]] = self.nthuser+1
        self._sgy.binhead[self._par["bin_ntrailer"]] = self.ntxtrail
        self._sgy.binhead[self._par["bin_byteoff"]] = self._fp.skip
        self._sgy.binhead[self._par["bin_iconst"]] = 16909060

        return self._sgy.binhead

    def log_binhead(self, binhead=None, zero=False):
        """
        Log the SEG-Y binary file header.

        Parameters
        ----------
        binhead : Numpy structured array, optional (default: None)
            The binary header. If 'None', the internally stored binary header
            (if available) is used.
        zero : boolean, optional (default: False)
            Whether to print binary header entries that are zero or not.
        """
        self._log_binhead(binhead=binhead, zero=zero)

    def log_txthead(self, txthead=None, info=None):
        """
        Log the (primary) textual file header.

        txthead : list of strings or string, optional (default: None)
            The textual header. If 'None', the internally stored textual
            header (if available) is used.
        info : str, optional (default: None)
            A verbal description of this textual header.
        """
        self._log_txthead(txthead=txthead, info=info)

    def init(self, **kwargs):
        """
        Initialize output and write all SEG-Y file headers to disk.

        This includes the standard textual header, the binary header, and
        possibly any extended header records. This function has to be called
        before writing any traces to the file.

        Parameters
        ----------
        textual : list of strings or string, optional (default: None)
            Primary textual file header (will be encoded 'ascii' or 'ebcdic').
            List of 40 strings. If a string is less than 80 characters, it will
            be padded up to the length of the card image. If a string is more
            than 80 characters, it will be truncated. If no textual header is
            supplied, a default template will be used. Alternatively, a single
            string of length 3200 bytes can be supplied.
        binary : Numpy structured array, optional (default: None)
            The SEG-Y binary header corresponding to the binary header
            definition file used in the class constructor. If no binary header
            is supplied, a default template with minimally pre-filled
            information based on parameters supplied to the class constructor
            will be used.
        records: string or list of strings, optional (default: None)
            The additional SEG-Y extended textual header(s). Each header record
            must be 3200-bytes long.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        """
        if self._head_written:
            log.warning("init() method called several times; call ignored.")
            return

        txth = kwargs.pop("textual", None)
        binh = kwargs.pop("binary", None)
        hrec = kwargs.pop("records", None)
        silent = kwargs.pop("silent", False)

        if kwargs:
            for key, val in kwargs.items():
                log.warning("Unknown argument '%s' with value '%s' encountered.", key, str(val))

        if txth is None:
            txth = self.txthead_template
            txth[0] = f"C01 This SEG-Y file was created by Python seisio version {__version__}."
            txth[0] = txth[0].ljust(_txtheader._SEGY_CARDLEN)
        self._sgy.txthead.header = txth
        # Should the last entries be checked for SEG-Y revision information
        # and 'END TEXTUAL HEADER' string? A lot of SEG-Y files do not provide
        # such entries although strictly speaking required by the standard.

        if binh is None:
            self._sgy.binhead = self.binhead_template
        else:
            self._sgy.binhead = binh.astype(self._sgy.bhdtype)

        if hrec is not None:
            if isinstance(hrec, str):
                if self.ntxtrec > 1:
                    raise ValueError("Received fewer ext. header records than expected. "
                                     "Check 'ntxtrec' parameter in constructor.")
                elif self.ntxtrec == 0:
                    log.warning("Expected no ext. header records. Ignoring 'records' parameter.")
                else:
                    self._sgy.txtrec[0].header = hrec
            else:
                if self.ntxtrec > len(hrec):
                    raise ValueError("Received fewer ext. header records than expected. "
                                     "Check 'ntxtrec' parameter in constructor.")
                elif self.ntxtrec < len(hrec):
                    log.warning("Received more ext. header records than expected (%d)."
                                "Ignoring add. 'records'.", self.ntxtrec)
                    for i in range(self.ntxtrec):
                        self._sgy.txtrec[i].header = hrec[i]
                else:
                    for i, rec in enumerate(hrec):
                        self._sgy.txtrec[i].header = rec
        elif self.ntxtrec > 0:
            raise ValueError(f"Expected {self.ntxtrec} extended header "
                             "record(s), received none.")

        with open(self._fp.file, "ab") as io:
            self._sgy.txthead.write(io)
            io.write(self._sgy.binhead.tobytes())
            for i in range(self.ntxtrec):
                self._sgy.txtrec[i].write(io)
            self._fp.filesize = io.tell()

        self._head_written = True
        if not silent:
            log.info("Wrote textual and binary file headers and %d add. header record(s).",
                     self.ntxtrec)

    def finalize(self, *args, encode=True, silent=False):
        """
        Finalize output (and write all SEG-Y file trailers to disk).

        This function has to be called after writing all traces to the file.
        Handle errors / inconsistencies gracefully to allow writing a file at
        the end of a job even if something seems wrong.

        Parameters
        ----------
        *args : 3200-byte buffer(s)
            The actual 3200-bytes trailer buffers.
        encode : bool, optional (default: True)
            ASCII-encode the trailers. If binary data are to be written as
            part of trailer stanzas, trailers should not be encoded.
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        """
        if self.ntxtrail > 0:
            if not isinstance(encode, bool):
                log.warning("Parameter 'encode' should be a boolean. Reset to 'True'.")
                encode = True

            if len(args) != self.ntxtrail:
                log.warning("Number of provided trailer records differs from expected number (%d).",
                            self.ntxtrail)
                if len(args) > self.ntxtrail:
                    log.warning("Some trailer records will be ignored.")
                else:
                    log.warning("Resetting number of trailer records to %d.", len(args))
                    del self._sgy.txtrail[len(args):]

            with open(self._fp.file, "ab") as io:
                for i, string in enumerate(args):
                    self._sgy.txtrail[i].set_header(string, encode=encode)
                    self._sgy.txtrail[i].write(io)

            log.info("Wrote %d trailer records.")

        self._sgy.binhead[self._par["bin_ntfile"]] = self._dp.nt
        self._sgy.binhead[self._par["bin_ntrailer"]] = self.ntxtrail

        log.info("Finalizing output file and re-writing updated binary header.")

        with open(self._fp.file, "r+b") as io:
            io.seek(self._sgy.txthead.size, 0)
            io.write(self._sgy.binhead.tobytes())
            io.seek(0, 2)
            self._fp.filesize = io.tell()

        log.info("Wrote a total of %d trace(s), file size: %d bytes.",
                 self._dp.nt, self._fp.filesize)

        self._tail_written = True
