import numpy as np
import numpy.typing as npt
import scipy.fftpack as fft


def bandpass_filter(
    input_data: npt.ArrayLike,
    lp_filter_resolution_ang: float,
    hp_filter_resolution_ang: float,
    pixel_size: float,
    filter_z_slice: bool = False,
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

    # input_data = np.array(input_data)

    if input_data.ndim != 3:
        log("Non 3D image detected! Disable z slice filtering.")
        filter_z_slice = False

    lp_filter_frequency = pixel_size / lp_filter_resolution_ang

    if filter_z_slice:
        old_shape = input_data.shape[-2:]
    else:
        old_shape = input_data.shape
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
    mask = mesh_dist <= (new_shape // 2 * 2 * lp_filter_frequency) ** 2

    if hp_filter_resolution_ang != 0:
        hp_filter_frequency = pixel_size / hp_filter_resolution_ang
        mask = (
            (new_shape // 2 * 2 * hp_filter_frequency) ** 2 <= mesh_dist
        ) & mask

    pad_list = []
    for shape_i in old_shape:
        diff_i = new_shape - shape_i
        pad_list.append([diff_i // 2, diff_i // 2 + diff_i % 2])

    if filter_z_slice:
        filtered_slices = []
        for cur_slice in input_data:
            pad_slice = np.pad(cur_slice, pad_list, "symmetric")
            pad_slice = fft.fftn(pad_slice)
            pad_slice = fft.fftshift(pad_slice)
            pad_slice *= mask
            pad_slice = fft.ifftshift(pad_slice)
            pad_slice = fft.ifftn(pad_slice)
            filtered_slices.append(pad_slice)
        pad_image = np.squeeze(np.stack(filtered_slices))

    else:
        pad_image = np.pad(input_data, pad_list, "symmetric")
        pad_image = fft.fftn(pad_image)
        pad_image = fft.fftshift(pad_image)
        pad_image *= mask
        pad_image = fft.ifftshift(pad_image)
        pad_image = fft.ifftn(pad_image)

    unpad_list = []
    for pad_i, shape_i in zip(pad_list, old_shape):
        unpad_list.append(np.s_[pad_i[0] : pad_i[0] + shape_i])

    return pad_image.real[tuple(unpad_list)], mask