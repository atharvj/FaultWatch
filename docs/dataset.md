# NASA C-MAPSS FD001

FaultWatch uses the 2008 Turbofan Engine Degradation Simulation Data Set from
the NASA Ames Prognostics Center of Excellence. FD001 simulates one operating
condition (sea level) and one high-pressure compressor degradation mode.
NASA publishes the dataset through its [PCoE repository, dataset 6](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and [NASA Open Data catalog](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

Retrieval was verified on 2026-11-23 using the [archive linked by NASA PCoE](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip).
The original package includes the October 2008 FD001 text files. The
[NASA-published Zenodo record v2](https://zenodo.org/records/15346912) contains
the byte-identical inner archive. NASA's Open Data catalog lists the license
as unspecified. The repository references the open download and credits NASA
while keeping the original archive and data files out of Git.

| File | SHA-256 |
|---|---|
| NASA PCoE outer archive | `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` |
| `CMAPSSData.zip` | `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f` |
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

```sh
uv run --locked faultwatch fetch
# If you downloaded the NASA PCoE or Zenodo archive separately:
uv run --locked faultwatch fetch --archive /path/to/CMAPSSData.zip
```

Both trajectory files have 26 whitespace-separated columns. They contain
engine ID, cycle, 3 operating settings and 21 sensors. The release readme
has an inconsistent sensor count on its last column line. The loader checks
the actual file widths, finite values, engine IDs and cycle continuity.

| Split | Rows | Engines | Cycles observed per engine |
|---|---:|---:|---:|
| Official train (run to failure) | 20,631 | 100 | 128–362 |
| Official test (truncated) | 13,096 | 100 | 31–303 |

Training RUL is the last observed training cycle minus the current cycle.
The test RUL vector contains one value per engine ID in ascending order and
applies to that engine's last observed cycle. Its 100 values range from 7 to
145 cycles. Engine IDs in the two files are separate namespaces.

Cite A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation
Simulation Data Set*, NASA Prognostics Data Repository, NASA Ames Research
Center. The bundled paper by Saxena, Goebel, Simon and Eklund (2008) describes
the damage-propagation simulation.
