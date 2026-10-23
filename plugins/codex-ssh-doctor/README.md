# SSH Doctor

Windows 本机使用的 Codex 本地插件，包含一个技能和确定性诊断脚本，无需额外 MCP 服务。

安装后在新聊天中选择 **SSH Doctor**，或使用 `$ssh-doctor`：

- “检查我的服务器 Codex 是否可用。”
- “检查 1455 端口是否冲突。”
- “重启笔记本代理任务，然后检查连接。”

默认只检查；只有明确提出重启时才操作现有 `Codex-Laptop-Proxy` 任务。不会自动重新登录、更改远端配置或发送模型请求。插件不会赋予额外的网络或执行权限。

## 安装

从本仓库根目录运行：

```powershell
codex plugin marketplace add .
```

重启/刷新 Codex 桌面客户端，在插件页找到本地市场 `llm-password-tools` 中的 **SSH Doctor** 并安装/启用。客户端也可通过用户配置启用：

```toml
[plugins."codex-ssh-doctor@llm-password-tools"]
enabled = true
```

需 Python 3.9+、Windows PowerShell、OpenSSH，目标服务器需 Python 3 与已安装的 Codex。HTTP 连通性检查使用服务器的 curl。本插件不负责安装代理、自启任务或服务器 Codex，原有安装脚本见仓库 `scripts/remote/`。

## 脱离插件直接运行

```powershell
python .\plugins\codex-ssh-doctor\skills\ssh-doctor\scripts\doctor.py check
python .\plugins\codex-ssh-doctor\skills\ssh-doctor\scripts\doctor.py check --host dr@10.137.11.194 --local-port 7890 --remote-port 17890
python .\plugins\codex-ssh-doctor\skills\ssh-doctor\scripts\doctor.py restart-proxy --confirm
```

第一版针对 Windows 笔记本 → Linux 服务器场景。可复制整个插件目录使用，脚本不依赖仓库根目录。默认主机和端口可用命令行参数覆盖。

返回 JSON，退出码 0 表示未发现失败项，1 表示发现失败，2 表示参数/授权条件不满足。警告和未检查仍可能存在，不应仅凭退出码 0 判断模型可用。默认检查不消耗模型额度。

检查本地代理、根目录计划任务、1455 端口、SSH、服务器代理 TCP/HTTPS 和 Codex 版本/登录。为避免读取进程环境中的密钥，不读取桌面后台进程环境，该项固定标记为未检查。SSH 失败后会把依赖它的检查标记为未检查。重启只定位任务计划程序提供的实际任务实例，并核对启动器及进程创建时间，不根据相似命令全局杀进程。

官方格式参考：[插件打包与本地市场](https://developers.openai.com/plugins/build/plugins)。
