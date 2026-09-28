from types import SimpleNamespace as NS

import pytest

from topodrome.site_agent import inventory as module


class GPM:
    NVML_GPM_METRICS_GET_VERSION = 1

    def __init__(self, values=(0.0, 0.0), error=0):
        self.values, self.error = values, error
        self.freed = []
        self.queries = 0

    def nvmlGpmQueryDeviceSupport(self, handle):
        return NS(isSupportedDevice=1)

    def nvmlGpmSampleAlloc(self):
        return object()

    def nvmlGpmSampleGet(self, handle, sample):
        pass

    def c_nvmlGpmMetricsGet_t(self):
        return NS(metrics=[NS(), NS()])

    def nvmlGpmMetricsGet(self, query):
        self.queries += 1
        for metric, value in zip(query.metrics, self.values, strict=True):
            metric.value, metric.nvmlReturn = value, self.error

    def nvmlGpmSampleFree(self, sample):
        self.freed.append(sample)


@pytest.mark.parametrize(
    "values,error,expected",
    [
        ((0.0, 0.0), 0, 0),
        ((0.1, 3.2), 0, 4),
        ((11.0, 0.0), 0, None),
        ((float("nan"), 0.0), 0, None),
        ((-1.0, 0.0), 0, None),
        ((0.0, 0.0), 3, None),
    ],
)
def test_independent_gpm_requires_valid_low_samples(values, error, expected):
    nvml = GPM(values, error)
    assert module._gpm_idle_utilization(nvml, 0, sleep=lambda _: None) == expected
    assert len(nvml.freed) == 2
    if expected is not None:
        assert nvml.queries == 2


def test_unsupported_gpm_never_claims_idle():
    assert module._gpm_idle_utilization(object(), 0) is None


def test_allocated_memory_excludes_driver_reservation():
    nvml = NS(nvmlMemory_v2=2, nvmlDeviceGetMemoryInfo=lambda h, version: NS(used=0))
    assert module._allocated_memory(nvml, 0, NS(used=521928704)) == 0
    assert module._allocated_memory(object(), 0, NS(used=521928704)) == 521928704


@pytest.mark.parametrize(
    "name,used,process,appears,expected",
    [
        ("RTX 5090", 0, False, False, 0),
        ("A800", 0, False, False, 100),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 0, False, False, 100),
        ("RTX 5090", 1024**3, False, False, 100),
        ("RTX 5090", 0, True, False, 100),
        ("RTX 5090", 0, False, True, 100),
    ],
)
@pytest.mark.parametrize("autodl", [False, True])
def test_inventory_narrow_fallback_and_process_recheck(
    monkeypatch, name, used, process, appears, expected, autodl
):
    calls = []
    nvml = NS(
        nvmlInit=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=lambda h: NS(total=32 * 1024**3, free=31 * 1024**3, used=used),
        nvmlDeviceGetUtilizationRates=lambda h: NS(gpu=100),
        nvmlDeviceGetName=lambda h: name,
        nvmlDeviceGetUUID=lambda h: "GPU-test",
    )
    provider = module.NVMLInventoryProvider(
        module.ProcessOwnershipRegistry(), autodl_gpm_fallback=autodl
    )
    provider._nvml = nvml

    def processes(*args):
        calls.append(1)
        return (NS(),) if process or (appears and len(calls) > 1) else ()

    monkeypatch.setattr(provider, "_processes", processes)
    monkeypatch.setattr(module, "_gpm_idle_utilization", lambda *args: 0)
    if autodl and used == 0 and not process and not appears:
        expected = 0
    assert provider.sample()[0].utilization_percent == expected
