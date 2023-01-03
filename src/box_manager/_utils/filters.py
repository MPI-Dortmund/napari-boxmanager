import numpy as np
import numpy.typing as npt
import scipy.fft as fft


def get_mask(
    lp_filter_resolution_ang: float,
    hp_filter_resolution_ang: float,
    pixel_size: float,
    input_shape,
) -> np.array:
    old_shape = input_shape
    new_shape = np.max(old_shape)
    box_range = np.arange(-new_shape // 2 + 1, 1 + new_shape // 2)

    mesh_dist = np.sum(
        np.array(
            np.meshgrid(*([box_range] * len(old_shape)), sparse=True),
            dtype=object,
        )
        ** 2,
        axis=0,
    )

    mask = mesh_dist >= 0
    if lp_filter_resolution_ang != 0:
        lp_filter_frequency = pixel_size / lp_filter_resolution_ang
        mask = (
            mesh_dist <= (new_shape // 2 * 2 * lp_filter_frequency) ** 2
        ) & mask

    if hp_filter_resolution_ang != 0:
        hp_filter_frequency = pixel_size / hp_filter_resolution_ang
        mask = (
            (new_shape // 2 * 2 * hp_filter_frequency) ** 2 <= mesh_dist
        ) & mask
    # mask = gaussian_filter(mask.astype(float), sigma=4)
    return mask


def bandpass_filter(
    input_data: npt.ArrayLike,
    lp_filter_resolution_ang: float,
    hp_filter_resolution_ang: float,
    pixel_size: float,
    log=print,
):

    if (
        lp_filter_resolution_ang >= hp_filter_resolution_ang
        and hp_filter_resolution_ang != 0
    ):
        log(
            f"{lp_filter_resolution_ang} cannot be greater than {hp_filter_resolution_ang}"
        )
        return None

    mask = get_mask(
        lp_filter_resolution_ang=lp_filter_resolution_ang,
        hp_filter_resolution_ang=hp_filter_resolution_ang,
        pixel_size=pixel_size,
        input_shape=input_data.shape,
    )

    pad_list = []
    for shape_i in input_data.shape:
        diff_i = np.max(input_data.shape) - shape_i
        pad_list.append([diff_i // 2, diff_i // 2 + diff_i % 2])

    pad_image = np.pad(input_data, pad_list, "symmetric")

    pad_image = fft.fftn(pad_image, workers=-1)
    pad_image = fft.fftshift(pad_image)

    pad_image *= mask
    pad_image = fft.ifftshift(pad_image)
    pad_image = fft.ifftn(pad_image, workers=-1)

    unpad_list = []

    for pad_i, shape_i in zip(pad_list, input_data.shape):
        unpad_list.append(np.s_[pad_i[0] : pad_i[0] + shape_i])
    filtered = pad_image.real[tuple(unpad_list)]
    return filtered, mask
