"""
desi_extract_spectra.py
Benjamin Floyd
2025-09-11

Provides two utility functions to retrieve spectra from the DESI database.
These functions were provided by Rahma Alfarsy and adapted by Benjamin Floyd.
"""

from pathlib import Path
from typing import LiteralString

import numpy as np
from astropy.table import Table
from desispec.coaddition import coadd_cameras
from desispec.io import read_spectra
import matplotlib.pyplot as plt
from desispec.spectra import Spectra, stack

_survey_type = LiteralString['main', 'sv1', 'sv2', 'sv3']
_program_type = LiteralString['bright', 'dark', 'backup', 'other']


def get_spectrum(target_ids: int | list[int], survey: _survey_type, program: _program_type, healpix: int,
                 specprod: str = 'loa', make_plot: bool = False) -> tuple[Spectra, plt.Figure] | Spectra:
    """
    Given a list of DESI Target IDs in the same HEALPix, survey, and program, create a merged 1D spectrum from the
    database.

    Parameters
    ----------
    target_ids:
        List of DESI Target IDs. Must be in the same HEALPix, survey, and program.
    survey:
        Survey from which to pull the spectra. Must be one of 'main', 'sv1', 'sv2', 'sv3'.
    program:
        Program to pull spectra from. Must be one of 'bright', 'dark', 'backup', 'other'.
    healpix:
        HEALPix containing the objects listed as Target ID.
    specprod:
        DESI catalog release name. Defaults to 'loa'.
    make_plot:
        Flag to produce a plot of the retrieved spectrum. Defaults to False.

    Returns
    -------
    (Spectra, plt.Figure) or Spectra
        Coadded 1D spectra and optionally accompanying figure.

    Raises
    ------
    FileNotFoundError
        Raised if file containing spectra cannot be found at data path location.

    Notes
    -----
    Adapted from original code courtesy of Rahma Alfarsy
    """

    # DESI sorts their HEALPix directories into groups
    healpix_grp = healpix // 100

    # Set the directory paths
    target_dir = Path(f'/dvs_ro/cfs/cdirs/desi/spectro/redux/{specprod}/'
                      f'healpix/{survey}/{program}/{healpix_grp}/{healpix}')
    spectra_filepath = target_dir / f'coadd-{survey}-{program}-{healpix}.fits'

    if spectra_filepath.exists():
        # Only read the FIBERMAP, WAVE, FLUX, IVAR HDUs. If other data is required edit the skip_hdus list.
        spectra = read_spectra(str(spectra_filepath), targetids=target_ids,)
                               # skip_hdus=['EXP_FIBERMAP', 'SCORES', 'EXTRA_CATALOG', 'MASK', 'RESOLUTION'])

        # Coadd the spectra across all cameras to create a merged 'brz' spectra
        spectra_camcoadd = coadd_cameras(spectra)

        if make_plot:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(spectra_camcoadd.wave['brz'], spectra_camcoadd.flux['brz'][0], lw=0.5, alpha=1)
            ax.axvspan(5660, 5930, alpha=0.1, color='k')
            ax.axvspan(7470, 7720, alpha=0.1, color='k')
            ax.set(xlabel='Wavelength [Angstrom]', ylabel=r'Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')

            return spectra_camcoadd, fig
        return spectra_camcoadd
    else:
        raise FileNotFoundError(f'coadd-{survey}-{program} not found.')

def get_epochs_spectra(target_id: int, survey: _survey_type, program: _program_type, healpix: int, specprod: str = 'loa',
                       make_plot: bool = False):
    """
    Given a specific DESI Target ID, retrieve spectra from all observation epochs that has been observed in a given
    survey

    Parameters
    ----------
    target_id:
        DESI Target ID
    survey:
        Survey from which to pull the spectra. Must be one of 'main', 'sv1', 'sv2', 'sv3'.
    program:
        Program to pull spectra from. Must be one of 'bright', 'dark', 'backup', 'other'.
    healpix:
        HEALPix containing the specified object.
    specprod:
        DESI catalog release name. Defaults to 'loa'.
    make_plot:
        Flag to produce a plot of the retrieved spectrum. Defaults to False.

    Returns
    -------


    Raises
    ------
    FileNotFoundError:
        Raised if file containing spectra cannot be found at data path location.

    Notes
    -----
    Adapted from original code courtesy of Rahma Alfarsy.

    """


    # coadds spectra taken on the same night

    spectra_camcoadds = []
    ds = []

    # DESI sorts their HEALPix directories into groups
    healpix_grp = healpix // 100

    # Set the directory paths
    target_dir = Path(f'/dvs_ro/cfs/cdirs/desi/spectro/redux/{specprod}/'
                      f'healpix/{survey}/{program}/{healpix_grp}/{healpix}')
    spectra_filepath = target_dir / f'spectra-{survey}-{program}-{healpix}.fits.gz'

    if spectra_filepath.exists():
        spectra = read_spectra(str(spectra_filepath), targetids=target_id,
                               skip_hdus=['EXP_FIBERMAP', 'SCORES', 'EXTRA_CATALOG', 'MASK', 'RESOLUTION'])

        # TODO Figure out what this is doing
        # List all observation dates for our Target ID truncating our MJDs to integers to reduce to the day
        mjds = np.array([spectra.fibermap['MJD']], dtype=int)

        # Reduce our dates to unique observation days
        dates = np.unique(mjds)
        ds.extend(dates)

        spectra_camcoadds = [coadd_cameras(stack(spectra[spectra.fibermap['MJD'] == date])) for date in dates]

        if make_plot:
            fig, ax = plt.figure(figsize=(5,4))
            for i in range(len(ds)):
                # For visual reference to the band overlap regions
                wavepad = 15
                wavemin_br = spectra_camcoadds[i].wave['r'].min() - wavepad
                wavemax_br = spectra_camcoadds[i].wave['b'].max() + wavepad
                wavemin_rz = spectra_camcoadds[i].wave['z'].min() - wavepad
                wavemax_rz = spectra_camcoadds[i].wave['r'].max() + wavepad

                ax.plot(spectra_camcoadds[i].wave["brz"], spectra_camcoadds[i].flux["brz"][0], lw=0.5, alpha=1)
                ax.axvspan(wavemin_br, wavemax_br, alpha=0.1, color='k')
                ax.axvspan(wavemin_rz, wavemax_rz, alpha=0.1, color='k')
            ax.set(xlabel='Wavelength [Angstrom]', ylabel=r'Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]')

            return spectra_camcoadds, ds, fig
        return spectra_camcoadds, ds
    else:
        raise FileNotFoundError(f'coadd-{survey}-{program} not found.')