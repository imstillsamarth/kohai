from kohai.advisor import CLOUD, OK, SLOW, advise
from kohai.catalog import load_catalog
from kohai.sysinfo import GPU, SystemInfo


def machine(gpus, ram=16.0):
    return SystemInfo(
        os="Windows",
        os_release="11",
        arch="AMD64",
        cpu="Test CPU",
        cores=8,
        threads=16,
        ram_gb=ram,
        disk_free_gb=200,
        python="3.12.0",
        in_wsl=False,
        gpus=gpus,
    )


def nvidia(vram):
    return GPU(name=f"Test GPU {vram}GB", vendor="nvidia", vram_gb=vram)


def test_catalog_loads_and_is_sorted():
    cat = load_catalog()
    assert len(cat) >= 10
    assert [m.params_b for m in cat] == sorted(m.params_b for m in cat)
    assert len({m.id for m in cat}) == len(cat)


def test_6gb_gpu_recommends_small_rl_student():
    a = advise(machine([nvidia(6.0)]), load_catalog())
    assert a.backend == "cuda"
    assert a.recommended is not None
    assert a.recommended.rl == OK
    assert 1.0 <= a.recommended.model.params_b <= 2.0
    assert a.settings["batch_size"] == 1


def test_big_gpu_recommends_bigger_model():
    small = advise(machine([nvidia(6.0)]), load_catalog())
    big = advise(machine([nvidia(24.0)]), load_catalog())
    assert big.recommended.model.params_b > small.recommended.model.params_b


def test_cpu_only_points_to_colab():
    a = advise(machine([]), load_catalog())
    assert a.backend == "cpu"
    assert a.train_where == "colab"
    assert all(v.rl == CLOUD for v in a.verdicts)
    assert any(v.train == SLOW for v in a.verdicts)
    assert a.recommended is not None


def test_amd_gpu_warns():
    a = advise(machine([GPU(name="AMD Radeon RX 7800", vendor="amd")]), load_catalog())
    assert a.backend == "cpu"
    assert any("NVIDIA" in w for w in a.warnings)
