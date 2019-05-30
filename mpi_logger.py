"""
mpi_logger.py
Author: Benjamin Floyd

Taken from https://gist.github.com/sixy6e/ed35ea88ba0627e0f7dfdf115a3bf4d1#file-mpi_logger-py
"""

import logging
from mpi4py import MPI


class MPIIOStream(object):

    """
    A very basic MPI stream handler for synchronised I/O.
    """

    def __init__(self, filename, comm, mode):

        self._file = MPI.File.Open(comm, filename, mode)
        self._file.Set_atomicity(True)

    def write(self, msg):
        # if for some reason we don't have a unicode string...
        try:
            msg = msg.encode()
        except AttributeError:
            pass
        self._file.Write_shared(msg)

    def sync(self):
        """
        Synchronise the processes
        """
        self._file.Sync()

    def close(self):
        self.sync()
        self._file.Close()


class MPIFileHandler(logging.StreamHandler):

    """
    A basic MPI file handler for writing log files.
    Internally opens a synchronised MPI I/O stream via MPIIOStream.
    Ideas and some code from:
    * https://groups.google.com/forum/#!topic/mpi4py/SaNzc8bdj6U
    * https://gist.github.com/JohnCEarls/8172807
    * https://stackoverflow.com/questions/45680050/cannot-write-to-shared-mpi-file-with-mpi4py
    """

    def __init__(self, filename,
                 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE, comm=MPI.COMM_WORLD):
        self.filename = filename
        self.mode = mode
        self.comm = comm

        super(MPIFileHandler, self).__init__(self._open())

    def _open(self):
        stream = MPIIOStream(self.filename, self.comm, self.mode)
        return stream

    def close(self):
        if self.stream:
            self.stream.close()
            self.stream = None

    def emit(self, record):
        """
        Emit a record.
        We have to override emit, as the logging.StreamHandler has 2 calls
        to 'write'. The first for the message, and the second for the
        terminator. This posed a problem for mpi, where a second process
        could call 'write' in between these two calls and create a
        conjoined log message.
        """
        msg = self.format(record)
        self.stream.write('{}{}'.format(msg, self.terminator))
        self.flush()