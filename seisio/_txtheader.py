"""A module to handle textual headers of SEG-Y files."""

import logging

log = logging.getLogger(__name__)

_SEGY_CARDLEN = 80
_SEGY_NUMCARD = 40

# SEGY textual header encoding
_TXT_ENCODING = {"cp037": "EBCDIC",
                 "ascii": "ASCII"}


class TxtHeader:
    """Class TxtHeader to handle textual headers and their encodings."""

    def __init__(self, encoding=None, info="SEG-Y textual file header"):
        """
        Initialize class TxtHead.

        Parameters
        ----------
        enconding : str, optional (default: None)
            Encoding of the textual header. Either 'ebcdic', 'ascii' or None.
            If None, the encoding will be determined automatically when
            reading a textual header. For writing headers, the encoding
            defaults to 'ascii' if None on input.
        info : str, optional (default: "SEG-Y textual file header")
            A verbal description of this textual header.
        """
        self.encoding = encoding
        self.info = info
        self.header = None              # encoded, single string
        self._bytes = None              # raw header
        self._list = None               # encoded, list of strings

    @property
    def size(self):
        """Return the size (in bytes) of this (textual) header."""
        return _SEGY_CARDLEN * _SEGY_NUMCARD

    def __len__(self):
        """Return the size (in bytes) of this (textual) header."""
        return self.size

    @property
    def info(self):
        """Return descriptive information about this (textual) header."""
        return self._info

    @info.setter
    def info(self, value):
        """Set the descriptive information for this (textual) header."""
        self._info = "Unknown" if value is None else value

    @property
    def encoding(self):
        """Return the encoding of this (textual) header."""
        if self._encoding is not None:
            return _TXT_ENCODING[self._encoding]
        return None

    @encoding.setter
    def encoding(self, value):
        """Set the encoding of this (textual) header."""
        if value is None:
            self._encoding = None
        elif isinstance(value, str):
            value = value.lower()
            if value not in ["ebcdic", "ascii"]:
                raise ValueError(f"Unknown value '{value}' for parameter 'encoding'.")
            self._encoding = "cp037" if value == "ebcdic" else "ascii"
        else:
            raise TypeError("Unknown type for parameter 'encoding'.")

    def _ascii_or_ebcdic(self):
        """
        Determine whether a buffer contains ASCII or EBCDIC.

        The code for a space character in ASCII is hex 20, there is no character
        in the EBCDIC alphabet that has the same code. However, the code for a
        space character in EBCDIC is hex 40, which in the ASCII alphabet is the
        code for the '@' symbol. The ASCII codes for the digits 0 through 9 are
        hex 30 to hex 39 which have no correspondence with any EBCDIC codes. The
        EBCDIC codes for the digits 0 through 9 are hex f0 to hex f9 which have
        no correspondence with any ASCII codes. Use this information to 'guess'
        whether a buffer is ASCII or EBCDIC.

        Returns
        -------
        str
            Encoding of this header. Either 'cp037' (EBCDIC) or 'ascii' (ASCII).
        """
        encoding = None
        n_ascsp = n_ebcsp = n_ascdig = n_ebcdig = 0

        for cbyte in self._bytes:
            if cbyte == 0x20:
                n_ascsp += 1
            elif cbyte == 0x40:
                n_ebcsp += 1
            else:
                if 0x30 <= cbyte <= 0x39:
                    n_ascdig += 1
                elif 0xf0 <= cbyte <= 0xf9:
                    n_ebcdig += 1

        if n_ascsp > n_ebcsp:
            # very likely ASCII
            encoding = "ascii"
            # Some SEG-Y appear to be written in EBCDIC but there are lots
            # of ASCII space characters and no EBCDIC space characters
            if n_ebcdig > n_ascdig:
                encoding = "cp037"
        elif n_ebcsp > n_ascsp:
            # very likely EBCDIC
            encoding = "cp037"
        else:
            if n_ascdig > n_ebcdig:
                # likely ASCII
                encoding = "ascii"
            elif n_ebcdig > n_ascdig:
                # likely EBCDIC
                encoding = "cp037"
            else:
                # no clue, go with tradition
                encoding = "cp037"

        log.info("%s encoding looks to be '%s' (best guess).", self._info, _TXT_ENCODING[encoding])

        return encoding

    def _to_list(self):
        """Transfer textual header to list of strings."""
        self._list = [self._header[i:i+_SEGY_CARDLEN] for i in
                      range(0, self.size, _SEGY_CARDLEN)]

    @property
    def header(self):
        """
        Return the textual header as list.

        Returns
        -------
        list
            The textual header. An empty list is returned if no decoded header
            is available.
        """
        if self._list is not None:
            return self._list
        return []

    @header.setter
    def header(self, header):
        """Set the encoded textual header."""
        self.set_header(header, encode=True)

    def get_header(self, decode=True):
        """
        Return the textual header as single string.

        Either the decoded string or the raw buffer can be returned. The raw
        buffer should be requested if it contains binary, which could be the
        case for any textual trailers. They need to be decoded by the user
        based on available information about that textual header and cannot
        be done automatically.

        Parameters
        ----------
        decode : bool, optional (default: True)
            If 'True', the decoded header is returned. If set to 'False', the
            raw header is returned.

        Returns
        -------
        str
            Textual header as single 3200-character long string.
        """
        if decode:
            if self._header is not None:
                return self._header
        else:
            if self._bytes is not None:
                return self._bytes
        return str("")

    def set_header(self, header, encode=True, silent=False):
        """
        Set the textual header.

        The header will be truncated or expanded as necessary to have the
        correct length.

        Parameters
        ----------
        header : str or list of str, or None
            3200-byte long string or list of max. 40 strings, each max. 80
            characters long.
        encode : boolean, optional (default: True)
            Convert (encode) header (True; default) or keep as is (False).
        silent : bool, optional (default: False)
            Whether to suppress all standard logging (True) or not (False).
        """
        if header is None:
            self._header = None
            return

        if isinstance(header, list):
            hdr = ""
            count = 0
            for count, line in enumerate(header):
                hdr = ''.join([hdr, line.ljust(_SEGY_CARDLEN)[:_SEGY_CARDLEN]])
            if count < _SEGY_NUMCARD-1:
                hdr.ljust(self.size)
        else:
            hdr = header.ljust(self.size)[:self.size]

        self._header = hdr

        if encode:
            if not silent:
                log.info("%s encoding set to '%s'.", self._info, _TXT_ENCODING[self._encoding])
            self._to_list()
            if self._encoding is None:
                self._encoding = "ascii"
            self._bytes = self._header.encode(self._encoding)
        else:
            self._bytes = self._header

    def read(self, fp):
        """
        Read the textual header from a file.

        Parameters
        ----------
        fp : I/O stream
            The file pointer must be correctly positioned before calling
            this function.
        """
        self._bytes = fp.read(self.size)

        if len(self._bytes) < self.size:
            log.error("Bytes read: %d, bytes expected: %d.",
                      len(self._bytes), self.size)
            raise EOFError(f"Short read of textual header '{self._info}'.")

        if self._encoding is None:
            self._encoding = self._ascii_or_ebcdic()
        else:
            log.info("%s encoding set to '%s'.", self._info, _TXT_ENCODING[self._encoding])

        try:  # try to decode; can fail if, e.g., trailer contains binary
            self._header = self._bytes.decode(self._encoding)
            self._to_list()
        except ValueError:  # decoding failed, store header as is
            self._header = self._bytes
            self._list = None

    def write(self, fp):
        """
        Write the textual header to a file.

        Parameters
        ----------
        fp : I/O stream
            The file pointer must be correctly positioned before calling
            this function.
        """
        fp.write(self._bytes)

    @classmethod
    def template(self, major_version=1, minor_version=0, fill=True):
        """
        Return a textual header template.

        This function returns a list of 40 strings, each string 80 characters
        long starting with Cxx, where xx is the line number. A user can fill
        the textual header strings as required and use it later on to write a
        SEG-Y file to disk. C39 specifies the SEG-Y revision (defaults to 1.0),
        C40 specifies 'END TEXTUAL HEADER' as required by the SEG-Y standard.

        Note that additional header records do not necessarily consist of 40
        strings with 80 characters each. They are treated as 3200-bytes long
        single strings. Additional header records or trailer stanzas should
        not use the 'Cxx' convention for starting lines - this is reserved for
        the SEG-Y primary textual header.

        Parameters
        ----------
        major_version : int, optional (default: 1)
            Major SEG-Y revision number.
        minor_version : int, optional (default: 0)
            Minor SEG-Y revision number.
        fill : boolean, optional (default: True)
            Fill the individual strings with Cxx convention (True) or not.

        Returns
        -------
        list
            A list of strings, 40 card images (strings), each of 80 characters.
        """
        header = []
        if fill:
            for row in range(_SEGY_NUMCARD-2):
                header.append(f"C{row+1:02d}".ljust(_SEGY_CARDLEN))
            header.append(f"C{_SEGY_NUMCARD-1:02d} SEG-Y_REV{major_version}."
                          f"{minor_version}".ljust(_SEGY_CARDLEN))
            header.append(f"C{_SEGY_NUMCARD:02d} END TEXTUAL HEADER".ljust(_SEGY_CARDLEN))
        else:
            for row in range(_SEGY_NUMCARD):
                header.append(" ".ljust(_SEGY_CARDLEN))
        return header

    def log_txthead(self):
        """Pretty-print a textual header."""
        if self._list is None:
            return
        log.info("%s:", self._info)
        ostr = "\n".join(self._list)
        log.info("%s\n%s", "-------- BEGIN --------", ostr)
        log.info("%s", "--------- END ---------")


if __name__ == "__main__":
    try:
        txt = TxtHeader(encoding="StrangeValue")
    except ValueError:
        log.info("Caught an exception of type ValueError (as expected).")
    try:
        txt = TxtHeader(encoding=42.0)
    except TypeError:
        log.info("Caught an exception of type TypeError (as expected).")
    txt = TxtHeader(encoding="ascii")
    txt = TxtHeader(encoding="AsCiI")
    txt = TxtHeader(encoding="ebcdic")
    myheader = txt.template()
    tmp = list(myheader[0])
    tmp[4:7] = "ABC"
    myheader[0] = "".join(tmp)
    tmp = list(myheader[37])
    tmp[4:7] = "XYZ"
    myheader[37] = "".join(tmp)
    txt.header = myheader
    ebcdic = txt.get_header(decode=False)
    converted = ebcdic.decode("cp037")
    asciihead = txt.get_header(decode=True)
    txt.log_txthead()
    log.info("Length of %s: %d bytes", txt.info, len(txt))
