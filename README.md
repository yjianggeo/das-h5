# das-h5

Command-line DAS (Distributed Acoustic Sensing) HDF5 processing toolkit — read raw `.h5` files, filter, export to SAC/NumPy, draw waterfall / wiggle / single-trace figures, run a six-figure PSD analysis suite, and convert traces to audio.
DAS（分布式声学传感）HDF5 数据命令行处理工具——读取原始 `.h5` 文件、滤波、导出 SAC/NumPy、绘制波形瀑布图 / wiggle 图 / 单道图、运行六图 PSD 分析套件，并把单道转换为声音。

**Workflow / 流程:** read h5 (build CSV database index) → window/channel select → detrend + filter → export (SAC / NumPy) **and/or** plot (waterfall / wiggle / single-trace / PSD / audio)
读取 h5（建立 CSV 数据库索引）→ 选时窗/道窗 → 去趋势 + 滤波 → 导出（SAC / NumPy）**和/或** 绘图（瀑布图 / wiggle / 单道 / PSD / 声音）

> Repackaged and reorganized from the original `DASh5.py` toolkit by **Hong Heting (洪鹤庭, "Hardy Hong")**. The original algorithms are preserved; this package adds a clean module layout, a `dash5` console entry point, synthetic-data tests, and one optional new feature — global (whole-array) normalization for section plots (`-ng`).
> 本包在 **洪鹤庭（Hardy Hong）** 的原始 `DASh5.py` 代码基础上整理、修改而来。原始算法逻辑予以保留；本包新增清晰的模块结构、`dash5` 命令行入口、合成数据测试，以及一项可选新功能——剖面图的整体（全局）归一化（`-ng`）。

🌐 **[English](#english)** | **[中文](#中文)**

---

<a name="english"></a>

## English

### Features

- **Three input models**: HDF5 (`-h5`), SAC directory (`-sac`), and NumPy array (`-npy`).
- **HDF5 database index**: scans a directory tree of `.h5` files once into a CSV index (`.db`), so later extractions are fast and resumable; data gaps are zero-filled automatically.
- **Export**: to SAC (with event/location SAC headers) or NumPy `.npy`; a low-memory streaming path (`-sof`) converts large h5 volumes directly to SAC.
- **Filtering**: zero-phase Butterworth `bandpass` / `bandstop` / `lowpass` / `highpass`, with configurable order; anti-alias `decimate`.
- **Section plots**: waterfall (`imshow`) and wiggle (`fill_between`) images, horizontal or vertical, with per-channel or global (`-ng`) amplitude normalization.
- **Single-trace plots**: waveform + spectrogram for one channel, or a batch over a channel range.
- **PSD suite**: Welch power-spectral-density with a six-figure analysis set (time/freq/channel views, PQLX-style statistics), with on-disk PSD caching (`.pkl`).
- **Audio**: convert a single DAS trace to an audible `.wav`.
- **Tested**: synthetic-data unit tests for the processing, PSD, and I/O modules — no real DAS files required.

### Install

```bash
git clone https://github.com/yjianggeo/das-h5.git
cd das-h5
pip install -e .[dev]
```

Requires Python ≥ 3.10. Dependencies: `numpy`, `scipy`, `obspy`, `pandas`, `h5py`, `matplotlib`, `tqdm`, `psutil`.

This installs a console command `dash5`. If the scripts directory is not on your `PATH`, the equivalent invocation is `python -m das_h5.cli ...`.

### Data layout

Point `-d` at a directory tree of DAS `.h5` files (it is walked recursively). The first run can build a CSV **database index** (`-odb name.db`) that later runs reuse (`-idb name.db`):

```
project/
├── h5_file/                 # raw DAS .h5 files (any nesting)
│   ├── 20230705_000000.h5
│   └── ...
├── h5_test.db               # generated CSV index (space-separated)
└── das.loc                  # optional channel-location file
```

**`*.db` index columns** (space-separated, no header):

```
h5file_name  time_label  starttime  endtime  delta  npts  channel_amount
```

**`das.loc` format** (space-separated, no header) — used to fill SAC headers on export:

```
channel  latitude  longitude  altitude
0        30.1234   130.5678   12.0
1        30.1235   130.5679   12.0
```

### Quick start

All examples use the `dash5` command (or `python -m das_h5.cli`). Times are relative seconds from the reference time `-T`; `-c`/`-C` are the start/end channels.

**1. h5 model — build the index and extract a window:**

```bash
dash5 -h5 -d h5_file/ -T 20230705174505 -b 0 -e 60 -c 0 -C 100 -odb h5_test.db
```

**2. Export to SAC** (reusing the index `-idb`, adding event lat/lon/depth and channel locations):

```bash
dash5 -h5 -d h5_file/ -idb h5_test.db -T 20230705174505 -b 0 -e 60 -c 0 -C 100 \
      -so sac_data/ -loc das.loc -ea 30 -eo 130 -ed 10
```

Use `-sof` instead of `-so` for the low-memory streaming export of large volumes (optionally with `-de` to decimate).

**3. Export to NumPy `.npy`:**

```bash
dash5 -h5 -d h5_file/ -idb h5_test.db -T 20230705174505 -b 0 -e 60 -c 0 -C 100 -no npy_data/
```

**4. sac model / npy model** — read back already-exported data with `-sac dir/` or `-npy file.npy` in place of `-h5 -d`.

### CLI reference

**Input & selection**

| Flag | Long | Meaning |
|---|---|---|
| `-h5` / `-sac` / `-npy` | `--h5_model` / `--sac_model` / `--npy_model` | Choose the input model |
| `-d` | `--datadir` | Data directory (h5/sac) or `.npy` file |
| `-idb` / `-odb` | `--input_db` / `--out_db` | Reuse / write the CSV database index |
| `-T` | `--reftime` | Reference time, e.g. `20230705174505` |
| `-b` / `-e` | `--starttime` / `--endtime` | Relative start/end seconds from `-T` (`e > b`) |
| `-c` / `-C` | `--startchannel` / `--endchannel` | Start / end channel |

**Export**

| Flag | Long | Meaning |
|---|---|---|
| `-so` / `-sof` | `--sac_out` / `--sac_out_fast` | Output SAC dir / streaming low-memory SAC dir |
| `-no` | `--npy_out` | Output NumPy dir |
| `-de` | `--decimate` | Decimation factor |
| `-loc` | `--location` | `das.loc` channel-location file |
| `-ea` / `-eo` / `-ed` | `--evla` / `--evlo` / `--evdp` | Event latitude / longitude / depth (SAC headers) |

**Filtering**

| Flag | Long | Meaning |
|---|---|---|
| `-bp` / `-bs` | `--bandpass` / `--bandstop` | Band corners in Hz, e.g. `-bp 1 50` |
| `-lp` / `-hp` | `--lowpass` / `--highpass` | Corner in Hz |
| `-od` | `--order` | Filter order (default 4) |
| `-zr` | `--zero_phase` | Zero-phase (forward+backward) filtering |

**Waterfall (`imshow`) & wiggle (`fill`) sections**

| Flag | Long | Meaning |
|---|---|---|
| `-imo` / `-ims` | `--imshow_out` / `--imshow_show` | Save / show waterfall figure |
| `-imc` / `-imn` | `--imshow_cmap` / `--imshow_norm` | Colormap / color-norm range, e.g. `-imn -0.5 0.5` |
| `-flo` / `-fls` | `--fill_out` / `--fill_show` | Save / show wiggle figure |
| `-flc` / `-flg` | `--fill_cmap` / `--fill_gain` | Fill colors (e.g. `r b`) / wave gain |
| `-ng` | `--norm_global` | Global (whole-array) amplitude normalization for section plots, instead of the default per-channel |
| `-v` | `--vertical` | Vertical orientation (channel on x-axis) |
| `-fz` / `-dpi` | `--fig_size` / `--fig_dpi` | Figure size / DPI |

**Single trace & audio**

| Flag | Long | Meaning |
|---|---|---|
| `-num` | `--channel_num` | Plot a single channel |
| `-sio` / `-sro` | `--single_out` / `--single_range_out` | Save one figure / batch over `-c`..`-C` |
| `-spe` / `-spn` | `--spec_lim` / `--spec_norm` | Spectrogram freq limits / color norm |
| `-sdo` | `--sound_outpath` | Write `.wav` for the chosen `-num` trace |
| `-sds` / `-sdg` / `-sdi` | `--sound_scale` / `--sound_gain` / `--sound_interp` | Playback rate scale / amplitude gain / interpolation |

**Examples:**

```bash
# Filtered waterfall image
dash5 -h5 -d h5_file/ -T 20230705174505 -b 0 -e 60 -c 0 -C 500 -bp 1 50 -od 4 \
      -dpi 300 -imc RdBu -imn -0.5 0.5 -imo h5_imshow.png

# Filtered wiggle image
dash5 -h5 -d h5_file/ -T 20230705174505 -b 0 -e 60 -c 0 -C 500 -bp 1 50 \
      -flc r b -flg 4 -flo h5_fillplot.png

# Single trace (waveform + spectrogram)
dash5 -h5 -idb h5_test.db -T 20230705174505 -b 0 -e 60 -bp 1 50 -num 99 -sio h5-trace-99.png

# Trace to audio (.wav)
dash5 -h5 -idb h5_test.db -T 20230618125746 -b -8 -e 8 -c 0 -C 600 -num 480 \
      -sdo sound/ -sds 4 -lp 1500 -sdi 2 -sdg 8000
```

### PSD six-figure suite

Activate the PSD model with the window flags `-wle` (segment length, s), `-wst` (step, s), and one or more frequency bands `-mif`/`-maf` (with `-fnum` selecting the active band):

```bash
dash5 -h5 -idb h5_test.db -T 20230618125701 -b 46 -e 56 -c 0 -C 2000 \
      -wle 0.2 -wst 0.1 -mif 1 50 -maf 100 500 -fnum 1 -fall ./fig/ -chs 480 -smh -dpi 200
```

| Flag | Figure |
|---|---|
| `-f1` | Channel–Time RMS (T–X) |
| `-f2` | Channel–Frequency (F–X) |
| `-f3` | Frequency–Time for one channel (`-chs`) |
| `-f4` | PSD[dB]–Time |
| `-f5` | PSD[dB]–Frequency statistics (PQLX-style, `-bis` bins) |
| `-f6` | PSD[dB]–Frequency for chosen channels (`-chr` range or `-chc` list) |
| `-fall` | Save all six figures to a directory |

Caching: `-opd dir/` writes the computed PSD to a `.pkl`; `-ipd dir/` reads it back (skips recomputation, so `-h5 -d` is not needed). Display tweaks: `-smh` (smooth), `-lg` (log frequency), `-pr` (period axis).

### Modules

| Module | Responsibility |
|---|---|
| `das_h5.io` | HDF5 reading + CSV database index (`h5_reader`), SAC/NumPy writing (`writer`), streaming h5→SAC export (`sac_export`) |
| `das_h5.process` | Detrend, normalization, IIR filtering (`filters`); anti-alias decimation and interpolation (`resample`) |
| `das_h5.viz` | Waterfall imshow (`waterfall`), wiggle fill (`wiggle`), single-trace + spectrogram (`trace`) |
| `das_h5.psd` | Welch PSD computation (`compute`); six-figure plotting suite (`plots`) |
| `das_h5.sound` | DAS trace → `.wav` audio (`audio`) |
| `das_h5.cli` | Argparse command-line orchestration (`dash5` entry point) |

### Algorithm notes

**Strain-rate conversion.** Raw HDF5 samples (phase change rate) are converted to strain rate in `read_h5` as:

```
strain_rate = raw · 107.9 · 1e-9 · frequency / GaugeLength
```

where `frequency` is the repetition / time sampling rate from the metadata (`OutputDataRate`, i.e. `1 / native delta`) and `GaugeLength` is read from the `Acquisition` group attributes. Across multiple files, time gaps and out-of-range edges are zero-padded.

**Welch PSD.** `das_psd` uses `scipy.signal.welch` per sliding window (Hann window, `nperseg` = window samples, `nfft = nextpow2(window/delta)`, one-sided, `density` scaling [V²/Hz]). Results are sliced to the requested frequency band(s), and `das_psd_type` derives the RMS/time/frequency/statistics arrays for the six figures.

**Single-trace spectrogram.** `viz.trace` computes the spectrogram via `matplotlib.mlab` and renders it alongside the filtered waveform.

### Running tests

```bash
pytest tests/ -v
```

8 tests use synthetic data (process filtering/decimation, Welch PSD peak location, HDF5 read/write round-trip) — no real DAS files required.

### License

MIT

---

<a name="中文"></a>

## 中文

### 功能特性

- **三种输入模型**：HDF5（`-h5`）、SAC 目录（`-sac`）、NumPy 数组（`-npy`）。
- **HDF5 数据库索引**：把一棵 `.h5` 目录树扫描一次，写成 CSV 索引（`.db`），之后的提取既快又可断点续用；数据缺口自动补零。
- **导出**：导出为 SAC（含事件/台站 SAC 头）或 NumPy `.npy`；低内存流式通道（`-sof`）可把超大 h5 数据直接转为 SAC。
- **滤波**：零相位 Butterworth `bandpass` / `bandstop` / `lowpass` / `highpass`，阶数可调；抗混叠 `decimate`。
- **剖面图**：波形瀑布图（`imshow`）与 wiggle 图（`fill_between`），可横向或纵向，幅度可按道归一化或整体归一化（`-ng`）。
- **单道图**：单个通道的波形 + 频谱图，或对一段道范围批量出图。
- **PSD 套件**：Welch 功率谱密度，配六图分析套件（时间/频率/通道视角、PQLX 风格统计），并支持 PSD 磁盘缓存（`.pkl`）。
- **声音**：把单个 DAS 道转换为可听的 `.wav`。
- **测试**：处理、PSD、I/O 模块均有合成数据单元测试，无需真实 DAS 文件即可运行。

### 安装

```bash
git clone https://github.com/yjianggeo/das-h5.git
cd das-h5
pip install -e .[dev]
```

需要 Python ≥ 3.10。依赖：`numpy`、`scipy`、`obspy`、`pandas`、`h5py`、`matplotlib`、`tqdm`、`psutil`。

安装后会提供命令行命令 `dash5`。若脚本目录不在 `PATH` 中，等价写法是 `python -m das_h5.cli ...`。

### 数据组织

把 `-d` 指向一棵 DAS `.h5` 目录树（会被递归遍历）。首次运行可生成 CSV **数据库索引**（`-odb name.db`），之后复用（`-idb name.db`）：

```
project/
├── h5_file/                 # 原始 DAS .h5 文件（可任意层级）
│   ├── 20230705_000000.h5
│   └── ...
├── h5_test.db               # 生成的 CSV 索引（空格分隔）
└── das.loc                  # 可选：通道位置文件
```

**`*.db` 索引列**（空格分隔，无表头）：

```
h5file_name  time_label  starttime  endtime  delta  npts  channel_amount
```

**`das.loc` 格式**（空格分隔，无表头）——导出 SAC 时用于填充台站头：

```
通道号  纬度  经度  高程
0      30.1234  130.5678  12.0
1      30.1235  130.5679  12.0
```

### 快速开始

示例均使用 `dash5` 命令（或 `python -m das_h5.cli`）。时间为相对参考时间 `-T` 的秒数；`-c`/`-C` 为起止道号。

**1. h5 模型 —— 建立索引并提取一段：**

```bash
dash5 -h5 -d h5_file/ -T 20230705174505 -b 0 -e 60 -c 0 -C 100 -odb h5_test.db
```

**2. 导出为 SAC**（复用索引 `-idb`，并加入事件经纬深和通道位置）：

```bash
dash5 -h5 -d h5_file/ -idb h5_test.db -T 20230705174505 -b 0 -e 60 -c 0 -C 100 \
      -so sac_data/ -loc das.loc -ea 30 -eo 130 -ed 10
```

数据量大时用 `-sof` 代替 `-so` 走低内存流式导出（可配 `-de` 抽稀）。

**3. 导出为 NumPy `.npy`：**

```bash
dash5 -h5 -d h5_file/ -idb h5_test.db -T 20230705174505 -b 0 -e 60 -c 0 -C 100 -no npy_data/
```

**4. sac 模型 / npy 模型** —— 用 `-sac dir/` 或 `-npy file.npy` 取代 `-h5 -d`，读回已导出的数据。

### 命令行参数

**输入与选择**

| 参数 | 全称 | 含义 |
|---|---|---|
| `-h5` / `-sac` / `-npy` | `--h5_model` / `--sac_model` / `--npy_model` | 选择输入模型 |
| `-d` | `--datadir` | 数据目录（h5/sac）或 `.npy` 文件 |
| `-idb` / `-odb` | `--input_db` / `--out_db` | 复用 / 写入 CSV 数据库索引 |
| `-T` | `--reftime` | 参考时间，如 `20230705174505` |
| `-b` / `-e` | `--starttime` / `--endtime` | 相对 `-T` 的起止秒（`e > b`）|
| `-c` / `-C` | `--startchannel` / `--endchannel` | 起始 / 结束道号 |

**导出**

| 参数 | 全称 | 含义 |
|---|---|---|
| `-so` / `-sof` | `--sac_out` / `--sac_out_fast` | SAC 输出目录 / 低内存流式 SAC 目录 |
| `-no` | `--npy_out` | NumPy 输出目录 |
| `-de` | `--decimate` | 抽稀因子 |
| `-loc` | `--location` | `das.loc` 通道位置文件 |
| `-ea` / `-eo` / `-ed` | `--evla` / `--evlo` / `--evdp` | 事件纬度 / 经度 / 深度（SAC 头）|

**滤波**

| 参数 | 全称 | 含义 |
|---|---|---|
| `-bp` / `-bs` | `--bandpass` / `--bandstop` | 频带角频率（Hz），如 `-bp 1 50` |
| `-lp` / `-hp` | `--lowpass` / `--highpass` | 角频率（Hz）|
| `-od` | `--order` | 滤波器阶数（默认 4）|
| `-zr` | `--zero_phase` | 零相位（正反双向）滤波 |

**瀑布图（`imshow`）与 wiggle（`fill`）剖面**

| 参数 | 全称 | 含义 |
|---|---|---|
| `-imo` / `-ims` | `--imshow_out` / `--imshow_show` | 保存 / 显示瀑布图 |
| `-imc` / `-imn` | `--imshow_cmap` / `--imshow_norm` | 色图 / 颜色归一化范围，如 `-imn -0.5 0.5` |
| `-flo` / `-fls` | `--fill_out` / `--fill_show` | 保存 / 显示 wiggle 图 |
| `-flc` / `-flg` | `--fill_cmap` / `--fill_gain` | 填充颜色（如 `r b`）/ 波形增益 |
| `-ng` | `--norm_global` | 剖面图整体（全局）幅度归一化，替代默认的按道归一化 |
| `-v` | `--vertical` | 纵向排列（通道在 x 轴）|
| `-fz` / `-dpi` | `--fig_size` / `--fig_dpi` | 图尺寸 / DPI |

**单道与声音**

| 参数 | 全称 | 含义 |
|---|---|---|
| `-num` | `--channel_num` | 绘制单个通道 |
| `-sio` / `-sro` | `--single_out` / `--single_range_out` | 保存一张图 / 对 `-c`..`-C` 批量出图 |
| `-spe` / `-spn` | `--spec_lim` / `--spec_norm` | 频谱图频率范围 / 颜色归一化 |
| `-sdo` | `--sound_outpath` | 为所选 `-num` 道写出 `.wav` |
| `-sds` / `-sdg` / `-sdi` | `--sound_scale` / `--sound_gain` / `--sound_interp` | 播放速率倍数 / 幅度增益 / 插值 |

**示例：**

```bash
# 滤波后的瀑布图
dash5 -h5 -d h5_file/ -T 20230705174505 -b 0 -e 60 -c 0 -C 500 -bp 1 50 -od 4 \
      -dpi 300 -imc RdBu -imn -0.5 0.5 -imo h5_imshow.png

# 滤波后的 wiggle 图
dash5 -h5 -d h5_file/ -T 20230705174505 -b 0 -e 60 -c 0 -C 500 -bp 1 50 \
      -flc r b -flg 4 -flo h5_fillplot.png

# 单道（波形 + 频谱图）
dash5 -h5 -idb h5_test.db -T 20230705174505 -b 0 -e 60 -bp 1 50 -num 99 -sio h5-trace-99.png

# 单道转声音（.wav）
dash5 -h5 -idb h5_test.db -T 20230618125746 -b -8 -e 8 -c 0 -C 600 -num 480 \
      -sdo sound/ -sds 4 -lp 1500 -sdi 2 -sdg 8000
```

### PSD 六图套件

用时窗参数 `-wle`（段长，秒）、`-wst`（步长，秒）以及一个或多个频带 `-mif`/`-maf`（用 `-fnum` 选择当前频带）激活 PSD 模型：

```bash
dash5 -h5 -idb h5_test.db -T 20230618125701 -b 46 -e 56 -c 0 -C 2000 \
      -wle 0.2 -wst 0.1 -mif 1 50 -maf 100 500 -fnum 1 -fall ./fig/ -chs 480 -smh -dpi 200
```

| 参数 | 图 |
|---|---|
| `-f1` | 通道–时间 RMS（T–X）|
| `-f2` | 通道–频率（F–X）|
| `-f3` | 单通道（`-chs`）频率–时间 |
| `-f4` | PSD[dB]–时间 |
| `-f5` | PSD[dB]–频率统计（PQLX 风格，`-bis` 分箱）|
| `-f6` | 所选通道（`-chr` 范围或 `-chc` 列表）PSD[dB]–频率 |
| `-fall` | 把六张图全部保存到目录 |

缓存：`-opd dir/` 把算好的 PSD 写为 `.pkl`；`-ipd dir/` 读回（跳过重算，因此无需 `-h5 -d`）。显示调节：`-smh`（平滑）、`-lg`（对数频率）、`-pr`（周期轴）。

### 模块

| 模块 | 职责 |
|---|---|
| `das_h5.io` | HDF5 读取 + CSV 数据库索引（`h5_reader`）、SAC/NumPy 写出（`writer`）、流式 h5→SAC 导出（`sac_export`）|
| `das_h5.process` | 去趋势、归一化、IIR 滤波（`filters`）；抗混叠抽稀与插值（`resample`）|
| `das_h5.viz` | 瀑布图 imshow（`waterfall`）、wiggle fill（`wiggle`）、单道 + 频谱图（`trace`）|
| `das_h5.psd` | Welch PSD 计算（`compute`）；六图绘制套件（`plots`）|
| `das_h5.sound` | DAS 单道 → `.wav` 声音（`audio`）|
| `das_h5.cli` | argparse 命令行编排（`dash5` 入口）|

### 算法说明

**应变率换算。** 原始 HDF5 采样（相位变化率）在 `read_h5` 中换算为应变率：

```
strain_rate = raw · 107.9 · 1e-9 · frequency / GaugeLength
```

其中 `frequency` 为重复频率 / 时间采样率，取自元数据 `OutputDataRate`（即 `1 / 原始 delta`）；`GaugeLength` 取自 `Acquisition` 组属性。跨多文件时，时间缺口与超界边缘以补零处理。

**Welch PSD。** `das_psd` 对每个滑动窗用 `scipy.signal.welch` 计算（Hann 窗，`nperseg` = 窗内样点，`nfft = nextpow2(窗长/delta)`，单边，`density` 归一化 [V²/Hz]）。结果按所需频带切片，`das_psd_type` 再派生出六张图所需的 RMS/时间/频率/统计数组。

**单道频谱图。** `viz.trace` 用 `matplotlib.mlab` 计算频谱图，与滤波后的波形并排绘制。

### 运行测试

```bash
pytest tests/ -v
```

8 个测试使用合成数据（处理滤波/抽稀、Welch PSD 峰值定位、HDF5 读写往返），无需真实 DAS 文件。

### 许可证

MIT
