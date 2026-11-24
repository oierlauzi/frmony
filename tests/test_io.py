"""Reading subtomograms and their metadata, and writing the answer back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import simulate, wedge_mask
from frmony.io.maps import write_map, write_spectrum
from frmony.io.registry import available, read_source, register
from frmony.io.star import read_star, write_star
from frmony.io.volumes import crop_spectrum, prepare, soft_ball
from frmony.io.wedge import (
    cone_weight,
    half_to_centred,
    relion_weight,
    wedge_weight,
)

# ------------------------------------------------------------------ wedges


@pytest.mark.parametrize('tilt', [30.0, 60.0, 89.0])
def test_a_wedge_leaves_a_wedge(tilt: float) -> None:
    weight = wedge_weight(32, tilt)
    assert weight.shape == (32,) * 3
    assert set(np.unique(weight)) <= {0.0, 1.0}
    # A wider tilt range measures more.
    assert weight.mean() > wedge_weight(32, tilt / 2).mean()
    # The missing part is a wedge about the tilt axis, so it is symmetric.
    np.testing.assert_allclose(weight, np.flip(weight, axis=1))


def test_a_cone_leaves_a_cone() -> None:
    weight = cone_weight(32, 60.0)
    # A cone is what tilting about two axes leaves, and it is smaller than
    # the wedge one axis leaves, so more of Fourier space is measured.
    assert weight.mean() > wedge_weight(32, 60.0).mean()
    # A cone is symmetric under any turn about z.
    np.testing.assert_allclose(weight, np.swapaxes(weight, 1, 2), atol=1e-6)


@pytest.mark.parametrize('angle', [0.0, -5.0, 91.0])
def test_a_degenerate_tilt_is_refused(angle: float) -> None:
    with pytest.raises(ValueError, match='measures'):
        wedge_weight(16, angle)
    with pytest.raises(ValueError, match='measures'):
        cone_weight(16, angle)


def test_the_half_spectrum_layout_round_trips(
    rng: np.random.Generator,
) -> None:
    """RELION stores the weight the way FFTW lays it out, not centred."""
    size = 16
    field = rng.normal(size=(size,) * 3)
    even = 0.5 * (field + np.roll(np.flip(field), 1, axis=(0, 1, 2)))

    half = np.fft.ifftshift(even)[:, :, : size // 2 + 1]
    np.testing.assert_allclose(half_to_centred(half), even, atol=1e-6)


def test_a_relion_weight_is_unpacked(rng: np.random.Generator) -> None:
    """Two half spectra stacked along z: the squared transfer, then the
    plain count of tilts."""
    size = 16
    transfer = np.abs(rng.normal(size=(size,) * 3))
    transfer = 0.5 * (transfer + np.roll(np.flip(transfer), 1, axis=(0, 1, 2)))
    half = np.fft.ifftshift(transfer)[:, :, : size // 2 + 1]
    multiplicity = np.ones_like(half) * 41.0
    stored = np.concatenate((half, multiplicity), axis=0)

    # RELION accumulates the square of what reached the particle, so the
    # amplitude wanted is its root.
    np.testing.assert_allclose(
        relion_weight(stored, 'sqrt'), np.sqrt(transfer), atol=1e-5
    )
    np.testing.assert_allclose(
        relion_weight(stored, 'identity'), transfer, atol=1e-5
    )
    assert set(np.unique(relion_weight(stored, 'none'))) <= {0.0, 1.0}


def test_a_two_dimensional_stack_says_so() -> None:
    """RELION 5 extracts stacks by default and writes no weight for them.

    Failing on a missing column would send someone hunting; saying what to
    do instead does not.
    """
    with pytest.raises(ValueError, match='3D pseudo-subtomograms'):
        relion_weight(np.zeros((7, 8, 5)))
    with pytest.raises(ValueError, match='half spectra'):
        relion_weight(np.zeros((8, 8, 8)))


def test_an_unknown_weight_transform_is_refused() -> None:
    with pytest.raises(ValueError, match='sqrt'):
        relion_weight(np.zeros((8, 4, 3)), 'cube-root')


# --------------------------------------------------------------- preparing


def test_preparing_clears_the_zero_frequency(
    rng: np.random.Generator,
) -> None:
    """Which is what makes a constrained correlation's mean subtraction
    free at every rotation."""
    volume = rng.normal(size=(24,) * 3) + 5.0
    data, _ = prepare(volume, wedge_weight(24, 60.0))
    assert abs(complex(data[12, 12, 12])) == 0.0


def test_preparing_crops_the_band(rng: np.random.Generator) -> None:
    volume = rng.normal(size=(32,) * 3)
    data, weight = prepare(volume, wedge_weight(32, 60.0), crop=16)
    assert data.shape == weight.shape == (16,) * 3
    assert data.dtype == np.complex64
    assert weight.dtype == np.float32


def test_cropping_keeps_the_middle() -> None:
    spectrum = np.arange(8**3, dtype=np.complex128).reshape((8,) * 3)
    kept = crop_spectrum(spectrum, 4)
    np.testing.assert_array_equal(kept, spectrum[2:6, 2:6, 2:6])
    np.testing.assert_array_equal(crop_spectrum(spectrum, 8), spectrum)
    with pytest.raises(ValueError, match='does not fit'):
        crop_spectrum(spectrum, 16)


def test_a_mask_is_applied_before_the_transform(
    rng: np.random.Generator,
) -> None:
    volume = rng.normal(size=(16,) * 3)
    weight = wedge_weight(16, 60.0)
    plain, _ = prepare(volume, weight)
    masked, _ = prepare(volume, weight, mask=soft_ball(16, 4.0))
    assert np.abs(masked).sum() < np.abs(plain).sum()


def test_a_soft_ball_is_soft() -> None:
    radius, falloff = 12.0, 0.2
    ball = soft_ball(32, radius, falloff=falloff)
    width = falloff * 32

    assert ball[16, 16, 16] == pytest.approx(1.0)
    assert ball[0, 0, 0] == pytest.approx(0.0)
    # One well inside, nothing at the radius, and in between on the edge.
    assert ball[16, 16, 16 + int(radius - width - 1)] == pytest.approx(1.0)
    assert ball[16, 16, 16 + int(radius)] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < ball[16, 16, 16 + int(radius - width / 2)] < 1.0


def test_preparing_refuses_mismatches(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match='cube'):
        prepare(np.zeros((4, 4, 8)), np.zeros((4, 4, 8)))
    with pytest.raises(ValueError, match='does not go with'):
        prepare(np.zeros((8,) * 3), np.zeros((4,) * 3))
    with pytest.raises(ValueError, match='shape of the volume'):
        prepare(np.zeros((8,) * 3), np.zeros((8,) * 3), mask=np.zeros((4,) * 3))


# -------------------------------------------------------------------- star


def _write_relion(
    directory: Path, rng: np.random.Generator, count: int, version: str
) -> Path:
    """A small RELION project, as the extraction job would leave it."""
    import pandas as pd
    import starfile

    from frmony.io.maps import write_map

    size = 16
    simulation = simulate(rng, count, size=size, tilt=60.0)
    wedge = wedge_mask(size, 60.0)
    half = np.fft.ifftshift(wedge)[:, :, : size // 2 + 1]
    stored = np.concatenate((half, np.ones_like(half)), axis=0)

    names, weights = [], []
    for index in range(count):
        volume = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(simulation.data[index]))
        ).real
        data_path = directory / f'particle_{index:03d}_data.mrc'
        weight_path = directory / f'particle_{index:03d}_weights.mrc'
        write_map(data_path, volume, 4.0)
        write_map(weight_path, stored, 4.0)
        names.append(data_path.name)
        weights.append(weight_path.name)

    table = pd.DataFrame(
        {
            'rlnImageName': names,
            'rlnCtfImage': weights,
            'rlnTomoName': ['tomo1'] * count,
            'rlnOpticsGroup': [1] * count,
        }
    )
    if version == 'relion5':
        for axis in 'XYZ':
            table[f'rlnCenteredCoordinate{axis}Angst'] = rng.normal(size=count)
    else:
        for axis in 'XYZ':
            table[f'rlnCoordinate{axis}'] = rng.uniform(0, 500, count)

    optics = pd.DataFrame({'rlnOpticsGroup': [1], 'rlnImagePixelSize': [4.0]})
    path = directory / 'particles.star'
    starfile.write({'optics': optics, 'particles': table}, path, overwrite=True)
    return path


@pytest.mark.parametrize('version', ['relion4', 'relion5'])
def test_a_relion_table_is_read(
    tmp_path: Path, rng: np.random.Generator, version: str
) -> None:
    path = _write_relion(tmp_path, rng, 4, version)
    metadata = read_star(path)

    assert len(metadata) == 4
    assert metadata.version == version
    assert metadata.pixel_size == pytest.approx(4.0)
    assert metadata.weights is not None
    assert all(Path(name).is_absolute() for name in metadata.data)
    assert all(Path(name).exists() for name in metadata.data)


def test_a_two_dimensional_table_is_refused(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    import starfile

    path = _write_relion(tmp_path, rng, 3, 'relion5')
    blocks = starfile.read(path, always_dict=True)
    blocks['particles']['rlnTomoSubTomosAre2DStacks'] = 1
    starfile.write(blocks, path, overwrite=True)

    with pytest.raises(ValueError, match='Re-extract'):
        read_star(path)


def test_a_table_without_a_pixel_size_says_so(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    import starfile

    path = _write_relion(tmp_path, rng, 3, 'relion4')
    blocks = starfile.read(path, always_dict=True)
    del blocks['optics']
    starfile.write(blocks, path, overwrite=True)

    with pytest.raises(ValueError, match='pixel size'):
        read_star(path)


def test_the_answer_is_written_beside_what_came_in(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    import starfile

    from frmony.geometry import matrix_to_euler, random_rotations

    path = _write_relion(tmp_path, rng, 5, 'relion5')
    metadata = read_star(path)
    rotations = random_rotations(rng, 5)
    shifts = rng.normal(size=(5, 3))

    out = tmp_path / 'out.star'
    write_star(out, metadata, rotations, shifts)
    written = starfile.read(out, always_dict=True)['particles']

    # What came in is still there.
    assert 'rlnTomoName' in written.columns
    assert 'rlnCenteredCoordinateXAngst' in written.columns
    # And what the alignment found has been added.
    rot, _, _ = matrix_to_euler(rotations)
    # A STAR file stores decimal text, so this is the format's precision
    # rather than the conversion's.
    np.testing.assert_allclose(
        written['rlnAngleRot'], np.degrees(rot), atol=1e-4
    )
    np.testing.assert_allclose(
        written['rlnOriginXAngst'], shifts[:, 0] * 4.0, atol=1e-4
    )


# ---------------------------------------------------------------- registry


def test_the_registry_knows_both_layouts() -> None:
    assert set(available()) >= {'relion', 'mrc'}


def test_a_star_file_is_read_as_relion(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    path = _write_relion(tmp_path, rng, 3, 'relion5')
    source = read_source(path)
    assert source.format == 'relion5'
    assert source.weights is not None
    assert source.pixel_size == pytest.approx(4.0)


def test_a_list_of_volumes_is_read(tmp_path: Path) -> None:
    for index in range(3):
        write_map(tmp_path / f'v{index}.mrc', np.zeros((8,) * 3), 1.0)
    listing = tmp_path / 'volumes.txt'
    listing.write_text('# a comment\nv0.mrc\nv1.mrc\n\nv2.mrc\n')

    source = read_source(listing)
    assert source.format == 'mrc'
    assert source.weights is None
    assert len(source.data) == 3
    assert all(Path(name).exists() for name in source.data)


def test_an_empty_listing_is_refused(tmp_path: Path) -> None:
    listing = tmp_path / 'empty.txt'
    listing.write_text('# nothing here\n')
    with pytest.raises(ValueError, match='names no volumes'):
        read_source(listing)


def test_an_unknown_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='not a format'):
        read_source(tmp_path / 'x.star', format='nonsense')


def test_a_format_can_be_added(tmp_path: Path) -> None:
    """New conventions are registered, not wired into a chain of ifs."""
    from frmony.io.registry import Source

    register(
        'invented',
        lambda path, root: Source(
            data=['a.mrc'], weights=None, pixel_size=2.5, format='invented'
        ),
    )
    source = read_source(tmp_path / 'anything', format='invented')
    assert source.pixel_size == pytest.approx(2.5)


# -------------------------------------------------------------------- maps


def test_a_map_carries_its_pixel_size(tmp_path: Path) -> None:
    """vitrio binds no header metadata, so maps go through mrcfile."""
    import mrcfile

    volume = np.arange(8**3, dtype=np.float32).reshape((8,) * 3)
    path = tmp_path / 'map.mrc'
    write_map(path, volume, 3.25)

    with mrcfile.open(str(path)) as handle:
        np.testing.assert_allclose(handle.data, volume)
        assert float(handle.voxel_size.x) == pytest.approx(3.25, abs=1e-4)


def test_a_spectrum_is_written_as_its_volume(tmp_path: Path) -> None:
    import mrcfile

    rng = np.random.default_rng(0)
    volume = rng.normal(size=(8,) * 3)
    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(volume)))

    path = tmp_path / 'from_spectrum.mrc'
    write_spectrum(path, spectrum, 1.5)
    with mrcfile.open(str(path)) as handle:
        np.testing.assert_allclose(handle.data, volume, atol=1e-5)


def test_a_root_finds_images_that_moved(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    """RELION writes image paths relative to the project, not to the table.

    A table copied away from its project cannot find its images, and the
    root is how it is told where they stayed.
    """
    images = tmp_path / 'images'
    images.mkdir()
    original = _write_relion(images, rng, 3, 'relion5')

    copied = tmp_path / 'copied.star'
    copied.write_text(original.read_text().replace(str(images) + '/', ''))

    beside = read_source(copied)
    assert not Path(beside.data[0]).exists()

    told = read_source(copied, root=images)
    assert all(Path(name).exists() for name in told.data)
    assert told.weights is not None
    assert all(Path(name).exists() for name in told.weights)


def test_a_root_applies_to_a_listing_too(tmp_path: Path) -> None:
    images = tmp_path / 'vols'
    images.mkdir()
    for index in range(2):
        write_map(images / f'v{index}.mrc', np.zeros((8,) * 3), 1.0)
    listing = tmp_path / 'volumes.txt'
    listing.write_text('v0.mrc\nv1.mrc\n')

    assert not Path(read_source(listing).data[0]).exists()
    told = read_source(listing, root=images)
    assert all(Path(name).exists() for name in told.data)


def test_a_root_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    listing = tmp_path / 'volumes.txt'
    listing.write_text('v0.mrc\n')
    with pytest.raises(ValueError, match='not a directory'):
        read_source(listing, root=tmp_path / 'nowhere')


def test_whitening_makes_the_noise_of_a_summed_particle_white() -> None:
    # What RELION stores is W S + n with a noise of variance W, where W
    # is the weight file. Build exactly that, with no signal, and the
    # whitened spectrum must have the same power wherever anything was
    # measured, however uneven the transfer. Measured: the ratio of the
    # power in the heaviest tenth to the lightest is 1.00 after and 30
    # before.
    from frmony.particles import whiten

    rng = np.random.default_rng(3)
    size = 24
    transfer = rng.uniform(0.05, 4.0, size=(size,) * 3).astype(np.float32)
    transfer[:, :, : size // 4] = 0.0
    noise = rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    stored = (np.sqrt(transfer) * noise).astype(np.complex64)

    spectrum, weight = whiten(stored, transfer)

    assert np.all(spectrum[transfer == 0.0] == 0.0)
    assert np.all(weight[transfer == 0.0] == 0.0)
    inside = weight > 0.0
    np.testing.assert_allclose(
        weight[inside], np.sqrt(transfer[inside]), rtol=1e-6
    )
    order = np.argsort(transfer[inside])
    power = np.abs(spectrum[inside][order]) ** 2
    before = np.abs(stored[inside][order]) ** 2
    tenth = len(power) // 10
    assert power[-tenth:].mean() / power[:tenth].mean() == pytest.approx(
        1.0, abs=0.15
    )
    assert before[-tenth:].mean() / before[:tenth].mean() > 10.0


def test_whitening_leaves_a_wedge_and_a_clean_volume_alone(
    rng: np.random.Generator,
) -> None:
    # One model for every input, with no branch on where a particle came
    # from, only holds if the inputs that were right already come through
    # untouched. A transfer of ones and zeros is its own square root.
    from frmony.particles import whiten

    size = 16
    stored = (
        rng.normal(size=(size,) * 3) + 1j * rng.normal(size=(size,) * 3)
    ).astype(np.complex64)
    for transfer in (
        wedge_mask(size, 60.0),
        np.ones((size,) * 3, dtype=np.float32),
    ):
        spectrum, weight = whiten(stored * transfer, transfer)
        np.testing.assert_array_equal(weight, transfer)
        np.testing.assert_array_equal(spectrum, stored * transfer)


def test_whitening_does_not_divide_by_next_to_nothing() -> None:
    # The division is bounded: nothing under a ten-thousandth of the
    # largest transfer is divided by, so a spectrum of ones cannot come
    # out larger than a hundred.
    from frmony.particles import whiten

    transfer = np.full((4,) * 3, 1.0, dtype=np.float32)
    transfer[0] = 1e-9
    transfer[1] = 4e-4
    stored = np.ones((4,) * 3, dtype=np.complex64)
    spectrum, weight = whiten(stored, transfer)
    assert np.all(spectrum[0] == 0.0)
    assert np.all(weight[0] == 0.0)
    assert np.all(np.isfinite(spectrum))
    assert np.abs(spectrum).max() <= 100.0
    np.testing.assert_allclose(np.abs(spectrum[1]), 50.0, rtol=1e-5)


def test_a_mirror_is_written_as_the_rotation_it_is_minus(
    tmp_path: Path,
) -> None:
    import pandas as pd
    import starfile

    from frmony.geometry import euler_to_matrix
    from frmony.io.star import MIRRORED, Metadata

    rotation = euler_to_matrix(0.3, 1.1, -0.7)
    metadata = Metadata(
        particles=pd.DataFrame({'rlnImageName': ['a.mrc', 'b.mrc']}),
        optics=None,
        data=['a.mrc', 'b.mrc'],
        weights=None,
        pixel_size=2.0,
        version='relion5',
    )
    write_star(
        tmp_path / 'out.star',
        metadata,
        np.stack((rotation, -rotation)),
        np.zeros((2, 3)),
    )
    written = starfile.read(tmp_path / 'out.star')
    angles = written[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    np.testing.assert_allclose(angles[0], angles[1])
    np.testing.assert_array_equal(written[MIRRORED].to_numpy(), [0, 1])


def test_no_mirror_column_unless_there_is_a_mirror(tmp_path: Path) -> None:
    import pandas as pd
    import starfile

    from frmony.io.star import MIRRORED, Metadata

    metadata = Metadata(
        particles=pd.DataFrame({'rlnImageName': ['a.mrc']}),
        optics=None,
        data=['a.mrc'],
        weights=None,
        pixel_size=2.0,
        version='relion5',
    )
    write_star(
        tmp_path / 'out.star', metadata, np.eye(3)[None], np.zeros((1, 3))
    )
    assert MIRRORED not in starfile.read(tmp_path / 'out.star').columns
