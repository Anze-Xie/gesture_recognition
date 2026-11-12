# 实时手势识别与人机交互教学系统

本项目使用 OpenCV 采集摄像头画面，使用 MediaPipe Hands 检测手部 21 个关键点，结合几何规则与 KNN 实现手势识别、音量/亮度交互演示、石头剪刀布游戏和样本采集实验。

本文针对当前 `gesture_recognition_demo.py` 的实际实现，提供 **Windows 11（x64）** 与 **本机 Apple Silicon Mac（arm64）** 的环境安装和使用说明。

> 当前版本有两处需要先了解的限制：Mac 的音量和亮度仅作界面模拟；KNN 模式的 `1/2/3` 采样键与模式切换冲突，目前只能通过界面采集类别 `0`。详情见“操作说明”和“已知限制”。

## 1. 项目文件

```text
gesture_recognition_demo/
├── gesture_recognition_demo.py           # 主程序，包含离线自检
├── requirement.txt                       # Windows/macOS 共用依赖清单
├── readme.md                             # 本文
├── AI手势识别课程设计内容.docx            # 课程资料
├── ai_gesture_recognition_course_guide.pdf # 课程资料
├── .venv/                                # 按本文安装后生成的虚拟环境
└── gesture_samples.csv                   # 采集样本后生成的数据文件
```

依赖文件名是单数 **`requirement.txt`**。主程序开头注释和缺失依赖报错中的 `requirements.txt`、`README_ENV_ZH.md` 是旧名称；其中 Python 3.12 的说明也与本文统一采用的 3.11 不同，请以本文件中的命令为准。

Windows 与 Mac 必须分别创建各自的 `.venv`。复制项目时复制源码、依赖清单、文档和需要保留的 CSV，不要跨系统复制虚拟环境目录。

## 2. 开发环境与依赖

### 2.1 环境要求

| 项目        | Windows 11                            | 本机 Mac                                 |
| --------- | ------------------------------------- | -------------------------------------- |
| 系统/架构     | Windows 11，Intel/AMD x64              | macOS，Apple Silicon arm64              |
| Python    | CPython 3.11，64 位，建议 3.11.9 或同系列更新补丁版 | Homebrew Python 3.11；本机检测到 3.11.18     |
| Python 路径 | 创建虚拟环境后使用 `.venv\Scripts\python.exe`  | 本机基础解释器 `/opt/homebrew/bin/python3.11` |
| 图形环境      | 本机桌面会话                                | 本机桌面会话                                 |
| 摄像头       | 内置或 USB 摄像头，允许桌面应用访问                  | 内置或 USB 摄像头，允许启动终端/IDE 访问              |
| 计算设备      | CPU 即可，无需 CUDA                        | CPU 即可，无需额外 GPU 配置                     |

本机环境记录于 2026-11-12：macOS 26.5.1、arm64；系统 `/usr/bin/python3` 是 3.9.6，因此下面明确使用 Homebrew 的 Python 3.11 创建环境。不要向系统 Python 安装本项目依赖。

本安装方案统一使用 Python 3.11。固定的 MediaPipe 版本提供该解释器的 Windows x64 与 macOS 安装包；不要直接改用 Python 3.13/3.14 或 Windows ARM64 原生解释器。[MediaPipe 0.10.21 发布文件](https://pypi.org/project/mediapipe/0.10.21/)

### 2.2 依赖用途

| 包                           | 固定版本            | 用途与安装范围                        |
| --------------------------- | --------------- | ------------------------------ |
| `numpy`                     | 1.26.4          | 几何计算、特征归一化、内置 KNN              |
| `opencv-contrib-python`     | 4.11.0.86       | 提供 `cv2`：摄像头采集、绘制、窗口与按键        |
| `mediapipe`                 | 0.10.21         | 提供当前代码需要的 `mp.solutions.hands` |
| `scikit-learn`              | 1.6.1           | KNN 训练与预测；代码有 NumPy 回退，本环境统一安装 |
| `jax` / `jaxlib`            | 0.4.30 / 0.4.30 | MediaPipe 的传递依赖，固定为匹配的 CPU 版本  |
| `pycaw`                     | 20240210        | 仅 Windows：调用系统音量接口             |
| `comtypes`                  | 1.4.10          | 仅 Windows：COM 接口支持             |
| `screen-brightness-control` | 0.24.2          | 仅 Windows：在受支持的显示器上调整亮度        |

版本选择说明：

- 当前代码使用 MediaPipe Solutions 接口；固定 MediaPipe 和 NumPy 1.x，避免直接升级后出现接口或二进制兼容问题。该版 MediaPipe 的依赖声明要求 `numpy<2`。[发布元数据](https://pypi.org/pypi/mediapipe/0.10.21/json)
- **只安装一个 OpenCV 发行包**。MediaPipe 依赖 `opencv-contrib-python`，它已经包含普通 OpenCV 模块；不要同时安装 `opencv-python`，也不要换成没有桌面窗口支持的 `headless` 版本。[OpenCV 包说明](https://pypi.org/project/opencv-contrib-python/4.11.0.86/)
- 固定 Pycaw 旧版是为了匹配代码中的 `GetSpeakers().Activate(...)` 调用。[Pycaw 20240210 官方示例](https://pypi.org/project/pycaw/20240210/)
- `requirement.txt` 中的系统标记会让 pip 在 Mac 上跳过三个 Windows 专用包。亮度库本身不支持 macOS。[亮度库支持平台](https://pypi.org/project/screen-brightness-control/0.24.2/)
- `scipy`、`protobuf`、`matplotlib` 等间接依赖由 pip 自动解析。该文件固定核心依赖及关键兼容依赖，未锁定所有间接依赖。代码中的 `argparse`、`csv`、`math`、`os` 等属于 Python 标准库，无需另装。

## 3. Windows 11 安装配置

以下命令在 **PowerShell** 中执行。将示例目录 `C:\Projects\gesture_recognition_demo` 替换为实际项目位置。

### 3.1 安装 Python 3.11 x64

从 [Python 官方 3.11.9 发布页](https://www.python.org/downloads/release/python-3119/) 获取 Windows 64 位安装程序，或使用已经安装的兼容 Python 3.11。安装时启用 pip、Python Launcher，并勾选将 Python 添加到 PATH。安装后重新打开 PowerShell，检查：

```powershell
py -3.11 --version
py -3.11 -c "import struct; print(struct.calcsize('P') * 8)"
```

应分别输出 `Python 3.11.x` 和 `64`。如果提示找不到 `py`，检查安装器中的 Python Launcher 选项；也可以用 Python 3.11 的完整路径替换下文的 `py -3.11`。

### 3.2 创建并安装虚拟环境

```powershell
cd C:\Projects\gesture_recognition_demo
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
.\.venv\Scripts\python.exe -m pip check
```

最后应显示 `No broken requirements found.`。这些命令直接使用虚拟环境解释器，不要求更改 PowerShell 执行策略。



或者利用conda 自行配置管理开发环境



如需在当前终端中直接使用 `python` 命令，可以激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止执行激活脚本，继续使用上面的完整解释器路径即可。本文 Windows 命令均不依赖激活。

### 3.3 验证并启动

```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy, mediapipe as mp, sklearn; print('OpenCV', cv2.__version__, 'NumPy', numpy.__version__, 'MediaPipe', mp.__version__, 'sklearn', sklearn.__version__); print(mp.solutions.hands.Hands)"
.\.venv\Scripts\python.exe gesture_recognition_demo.py --self-test
.\.venv\Scripts\python.exe gesture_recognition_demo.py
```

在 **设置 → 隐私和安全性 → 摄像头** 中开启摄像头访问，并允许桌面应用访问。关闭可能占用摄像头的会议软件后重试。

更换摄像头：

```powershell
.\.venv\Scripts\python.exe gesture_recognition_demo.py --camera 1
```

Windows 上模式 2 会尝试真实修改系统音量和第一个可控制显示器的亮度；能否成功取决于音频设备、显示器和驱动支持。

## 4. 本机 macOS 安装配置

以下命令在 Terminal、iTerm 或 IDE 的终端中执行。

### 4.1 检查 Python

本机已安装 Homebrew Python 3.11，直接执行：

```bash
/opt/homebrew/bin/python3.11 --version
/opt/homebrew/bin/python3.11 -c "import platform; print(platform.machine())"
```

本机预期为 `Python 3.11.18`、`arm64`。在其他 Apple Silicon Mac 上，如尚未安装 Python 3.11，可先通过 [Homebrew](https://brew.sh/) 安装 Homebrew，再运行：

```bash
brew install python@3.11
```

### 4.2 创建并安装虚拟环境

```bash
cd /Users/yongge/Desktop/gesture_recognition_demo
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python -m pip check
```

安装时看到 `Ignoring pycaw`、`Ignoring comtypes`、`Ignoring screen-brightness-control` 是正常现象，表示跳过 Windows 专用库。

**本机实测 `pip check` 会报告 `mediapipe 0.10.21 is not supported on this platform` 并返回非零状态。** 当前 universal2 安装包内的 WHEEL 标记仅写了 x86_64，但实际二进制包含 arm64；本机已通过模块导入、离线自检及 Hands 空白图像推理。若只有这一条提示，继续完成下面的运行验证；若还有缺失依赖或版本冲突，应先处理。不要把所有 `pip check` 错误都当作这一情况。

### 4.3 验证并启动

```bash
python -c "import cv2, numpy, mediapipe as mp, sklearn; print('OpenCV', cv2.__version__, 'NumPy', numpy.__version__, 'MediaPipe', mp.__version__, 'sklearn', sklearn.__version__); print(mp.solutions.hands.Hands)"
python gesture_recognition_demo.py --self-test
python gesture_recognition_demo.py
```

首次导入可能出现 `Matplotlib is building the font cache`，等待缓存生成即可。

首次启动时允许摄像头访问；也可前往 **系统设置 → 隐私与安全性 → 相机**，允许实际启动程序的终端或 IDE 使用相机。更改权限后重新打开该应用再启动程序。

更换摄像头：

```bash
python gesture_recognition_demo.py --camera 1
```

**当前 Mac 版本的模式 2 只更新界面的音量/亮度进度条。** 源码注释提到 `osascript`，但实际没有实现对应调用；安装更多 Python 库不会自动补齐这部分功能。

结束后退出虚拟环境：

```bash
deactivate
```

### 4.4 下次启动

无需重复安装依赖。重新打开终端后执行：

```bash
cd /Users/yongge/Desktop/gesture_recognition_demo
source .venv/bin/activate
python gesture_recognition_demo.py
```

也可以不激活环境，直接运行 `.venv/bin/python gesture_recognition_demo.py`。

## 5. IDE 开发配置

在 VS Code 或 PyCharm 中打开项目文件夹，将项目 Python 解释器设置为：

| 环境         | 项目解释器                                                             |
| ---------- | ----------------------------------------------------------------- |
| Windows 11 | 项目目录下的 `.venv\Scripts\python.exe`                                 |
| 本机 Mac     | `/Users/yongge/Desktop/gesture_recognition_demo/.venv/bin/python` |

运行文件选择 `gesture_recognition_demo.py`，工作目录设为项目根目录。程序将样本 CSV 写入**启动时的工作目录**；统一工作目录可以避免误以为数据丢失。自检可在运行参数中填写 `--self-test`，摄像头选择可填写 `--camera 1`。

## 6. 项目操作说明

### 6.1 启动参数

| 参数              | 默认值   | 作用                          |
| --------------- | ----- | --------------------------- |
| 不带参数            | 摄像头 0 | 打开实时识别窗口，初始进入模式 1           |
| `--camera N`    | `0`   | 使用索引为 N 的摄像头，如 `--camera 1` |
| `--self-test`   | 关闭    | 运行离线算法自检，不打开摄像头，不调整硬件       |
| `-h` / `--help` | —     | 查看命令帮助                      |

程序请求 1280×720 画面，实际分辨率取决于摄像头；画面水平镜像，最多检测两只手。点击视频窗口使其获得焦点，并使用英文输入法；字母按键使用小写。

### 6.2 快捷键

| 按键          | 当前实际行为                   |
| ----------- | ------------------------ |
| `1`         | 切换至几何规则识别                |
| `2`         | 切换至音量/亮度控制演示             |
| `3`         | 切换至石头剪刀布游戏               |
| `4`         | 切换至 KNN 模式               |
| 空格          | 在模式 3 开始新一轮 3 秒倒计时       |
| `0`         | 在模式 4、有手被检测到时记录一条类别 0 样本 |
| `t`         | 在模式 4 训练或重新训练 KNN        |
| `q` / `Esc` | 退出程序并释放摄像头               |

**当前模式 4 中，`1/2/3` 仍会切换模式，不能采样；`c` 没有对应处理逻辑。** 不要依照源码旧注释执行“按 c 后按 0–3 采集”的流程。

### 6.3 模式 1：几何规则手势识别

按 `1`，让手掌完整出现在画面中。界面显示骨架、手势名称、五指伸直状态、食指关节角度和 FPS。

支持规则包括：拳头/0、1、剪刀/2、3、4、张掌/5、OK、点赞、6/Call Me、Spider-Man/Love，以及其他五指组合的 `Custom` 标签。五指状态顺序为 **拇指、食指、中指、无名指、小指**，`1` 表示伸直，`0` 表示弯曲。捏合距离用于交互和 OK 判断，当前没有独立输出的 `Pinch` 分类标签。

保持光照充足、手部完整入镜，减少遮挡和快速运动。规则基于二维关键点几何关系，侧掌、遮挡等情况可能误判。

### 6.4 模式 2：音量与亮度交互

按 `2`，通过拇指与食指间距操作：

- 被识别为右手时，控制音量进度条；左手时，控制亮度进度条。
- 指尖间距约 25–180 像素映射到音量 0–100%、亮度 10–100%，并经过 EMA 平滑。
- 映射使用像素距离，手离摄像头的远近也会影响结果。

| 平台         | 音量                 | 亮度                               |
| ---------- | ------------------ | -------------------------------- |
| Windows 11 | Pycaw 初始化成功时尝试真实控制 | 亮度库和显示器支持时尝试真实控制，目标为 `display=0` |
| 本机 macOS   | HUD 界面模拟           | HUD 界面模拟                         |

硬件初始化或写入失败时，画面进度条仍可能变化。**进度条变化不等于系统设置已变化**，可对照系统音量/亮度验证。

### 6.5 模式 3：石头剪刀布

按 `3`，再按空格启动倒计时。在 3 秒结束时保持一种手势：握拳为石头、伸出食指和中指为剪刀、张开手掌为布。电脑随机出招并统计当前会话比分；再次按空格开始下一轮。

建议只展示一只手，倒计时结束时务必保持手部入镜。当前无手时可能被误当作石头；双手同时入镜时，判定采用最后处理的一只手。

### 6.6 模式 4：KNN 采样与训练

KNN 使用每只手 21 个点的 x/y 坐标，减去手腕位置并做尺度归一化，得到 42 维特征；不使用 z 坐标。默认类别编号为：

| 编号  | 默认显示名称                     | 建议手势 |
| --- | -------------------------- | ---- |
| 0   | `Class 0 (Fist/Stop)`      | 握拳   |
| 1   | `Class 1 (Peace/Victory)`  | 剪刀手  |
| 2   | `Class 2 (Open Palm)`      | 张开手掌 |
| 3   | `Class 3 (Custom Gesture)` | 自定义  |

当前可执行的流程：

1. 按 `4` 进入 KNN 模式，保持一只手入镜。
2. 展示类别 0 的手势，反复按 `0`，每次记录一条样本。
3. 总样本数达到 3 条后按 `t`，可以验证采集、保存、训练流程。
4. 仅有类别 0 时，模型无法区分多个手势，百分比也不能代表实际识别能力。
5. 若已有包含多个类别的兼容 CSV，放到工作目录并命名为 `gesture_samples.csv`，重启程序加载，进入模式 4 查看分类。完整的界面多类别采集需要先修复按键冲突。

训练使用距离加权 KNN，默认 `k=5`，实际取 `min(5, 样本数)`。首次加载 CSV 后，有至少 5 条样本会自动训练；只有 3–4 条时可按 `t` 手动训练。新增样本后需再次按 `t` 更新模型。

### 6.7 样本数据管理

- 默认文件为工作目录中的 `gesture_samples.csv`，每次采集追加一行，不保存视频或图片。
- CSV **无表头**；每行 43 列：第一列为类别编号，后面 42 列为归一化后的 `x0,y0,x1,y1,...,x20,y20`，保留 5 位小数。
- 程序保留 CSV，不单独保存模型文件；重启后从 CSV 重新训练。
- 清空实验数据前先退出程序，备份或重命名 CSV，再启动即可开始新的采集。
- CSV 读写异常当前会被静默忽略。采样后应检查文件是否存在、行数是否增加，不应只依赖界面的样本计数。
- KNN 显示的百分比来自邻居投票权重/分类概率，不能当作独立测试集上的识别准确率。

## 7. 环境检查与故障排查

以下命令中的 `python` 指项目虚拟环境解释器：Mac 可先激活 `.venv`；Windows 未激活时请替换为 `.\.venv\Scripts\python.exe`。

### 7.1 找不到模块、包装到了其他环境

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip install -r requirement.txt
python -m pip check
```

解释器和 pip 的路径都应指向项目 `.venv`。不要使用系统 `pip` 或 `sudo pip` 安装项目依赖。

### 7.2 `mediapipe` 没有 `solutions` 属性

确认当前环境安装的是固定版本，并确认项目中没有自己创建的 `mediapipe.py` 或同名目录遮蔽第三方包：

```bash
python -c "import mediapipe as mp; print(mp.__version__, mp.__file__); print(hasattr(mp, 'solutions'))"
python -m pip install -r requirement.txt
```

应输出 MediaPipe `0.10.21`，以及 `True`。

### 7.3 OpenCV 无法显示窗口，或 NumPy/二进制兼容错误

在**项目虚拟环境**中移除冲突的 OpenCV 包并按清单重装：

```bash
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
python -m pip install -r requirement.txt
python -m pip check
```

本程序需要本机桌面窗口，不适合直接在无图形界面的远程终端运行。若旧环境混装了大量包，建议用 Python 3.11 新建独立虚拟环境。

### 7.4 `No matching distribution found` / 安装时尝试源码编译

检查 Python 是否为 3.11、解释器架构是否匹配（Windows x64、本机 Mac arm64），并升级虚拟环境中的 pip。Apple Silicon 上避免混用 Rosetta x86_64 Python 和 arm64 环境。网络或软件源不可达时，恢复网络后重试；不要为绕过安装失败随意删除版本约束。

### 7.5 摄像头打不开、画面停住或黑屏

检查摄像头连接与系统权限，关闭占用相机的应用，尝试 `--camera 1` 或 `--camera 2`。程序收到读取失败会结束；离线 `--self-test` 成功不能证明摄像头正常。

若 Mac 提示 `Could not create an NSOpenGLPixelFormat`，请从本机桌面会话中的普通终端运行；受限沙箱或无图形会话可能阻止 MediaPipe 初始化图形上下文，即使推理本身使用 CPU。

### 7.6 按键没反应或采样时切换了模式

点击视频窗口，使其获得焦点，切换英文输入法。`q`、`t` 使用小写。模式 4 中 `1/2/3` 切换模式属于当前程序的按键冲突，不是依赖安装问题；`c` 当前无效。

### 7.7 Windows 音量/亮度不变

先检查可选硬件库是否成功导入：

```powershell
.\.venv\Scripts\python.exe -c "import pycaw, comtypes, screen_brightness_control as sbc; print(sbc.list_monitors())"
```

确认系统有可用音频输出设备，显示器及驱动支持亮度控制。某些外接显示器需要在显示器菜单中启用 DDC/CI；不受支持的设备可能只能显示界面效果。Mac 当前的模拟行为属于实现限制。

## 8. 已知限制与验证范围

- **已有代码限制**：KNN 多类别采样键冲突、`c` 键未实现；macOS 实际音量/亮度控制未实现。本文按实际代码记录，未修改这些功能。
- **游戏与采样**：建议使用单手；无手时游戏可能误判为石头，多手状态下使用最后处理的手。
- **自检覆盖范围**：`--self-test` 使用合成关键点，包含几何规则断言及平滑/KNN 流程演示，不覆盖摄像头、真实手部识别、键盘交互或硬件控制。
- **自检数据路径**：当前自检硬编码 `/tmp/test_gesture_samples.csv`。Windows 中该目录可能不存在，写入异常会被忽略，自检仍可能显示成功；请通过真实采样核实项目 CSV 的写入。
- **界面文字**：OpenCV 内置字体不能完整显示中文，部分中文游戏提示可能出现乱码。

### 8.1 本机验证记录（2026-11-12）

在本机 macOS 26.5.1 / arm64 / Python 3.11.18 的临时虚拟环境中验证：

| 检查项                                    | 结果                                           |
| -------------------------------------- | -------------------------------------------- |
| 按 `requirement.txt` 全新安装               | 成功；Windows 专用包自动跳过                           |
| NumPy、OpenCV、MediaPipe、scikit-learn 导入 | 成功；`mp.solutions.hands.Hands` 可用             |
| OpenCV 发行包检查                           | 仅安装 `opencv-contrib-python`                  |
| 程序 `--self-test`                       | 通过；输出 `All Self-Tests Passed Successfully!`  |
| Hands 引擎初始化与空白 RGB 图像处理                | 在允许图形上下文的本机进程中通过，无需摄像头                       |
| `pip check`                            | 未报告依赖版本冲突；报告上述 MediaPipe WHEEL 平台标记问题，退出码为 1 |
| 摄像头、GUI 按键、真实硬件控制                      | 未进行实机交互测试                                    |

验证安装在临时目录中完成；项目 `.venv` 需按第 3 或第 4 节自行创建。Windows 已核对核心包的 CPython 3.11 x64 发布文件与 Pycaw 调用接口，仍需在目标机器上完成安装、摄像头和硬件实测。

启动虚拟环境
conda activate D:\github_repositories\gesture_recognition\.venv