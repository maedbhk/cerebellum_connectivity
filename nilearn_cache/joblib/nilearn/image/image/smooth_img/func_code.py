# first line: 244
def smooth_img(imgs, fwhm):
    """Smooth images by applying a Gaussian filter.

    Apply a Gaussian filter along the three first dimensions of `arr`.
    In all cases, non-finite values in input image are replaced by zeros.

    Parameters
    ----------
    imgs : Niimg-like object or iterable of Niimg-like objects
        Image(s) to smooth (see
        http://nilearn.github.io/manipulating_images/input_output.html
        for a detailed description of the valid input types).

    fwhm : scalar, :class:`numpy.ndarray`, 'fast' or None
        Smoothing strength, as a Full-Width at Half Maximum (FWHM), in
        millimeters.
        If a scalar is given, width is identical on all three
        directions. A :class:`numpy.ndarray` must have 3 elements, giving the FWHM
        along each axis.
        If `fwhm='fast'`, a fast smoothing will be performed with
        a filter [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the scale.
        If `fwhm` is None, no filtering is performed (useful when just removal
        of non-finite values is needed).

        In corner case situations, `fwhm` is simply kept to None when `fwhm` is
        specified as `fwhm=0`.

    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image` or list of
        Filtered input image. If `imgs` is an iterable, then `filtered_img` is a
        list.

    """

    # Use hasattr() instead of isinstance to workaround a Python 2.6/2.7 bug
    # See http://bugs.python.org/issue7624
    if hasattr(imgs, "__iter__") \
       and not isinstance(imgs, str):
        single_img = False
    else:
        single_img = True
        imgs = [imgs]

    ret = []
    for img in imgs:
        img = check_niimg(img)
        affine = img.affine
        filtered = _smooth_array(get_data(img), affine, fwhm=fwhm,
                                 ensure_finite=True, copy=True)
        ret.append(new_img_like(img, filtered, affine, copy_header=True))

    if single_img:
        return ret[0]
    else:
        return ret
