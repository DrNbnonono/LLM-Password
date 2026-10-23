# 内网服务器使用 Codex

已验证环境（2026-10-23）：Windows 笔记本的 HTTP 代理 `127.0.0.1:7890`，服务器 `dr@10.137.11.194`，Ubuntu 24.04 x86_64，nvm Node.js 24.21.0，Codex CLI 0.164.0。

## 通道和边界

笔记本主动通过 SSH 建立反向转发：服务器 `127.0.0.1:17890` → 笔记本 `127.0.0.1:7890`。无需开放笔记本代理的局域网访问。仅使用代理环境变量的进程走这个出口，不改变服务器默认路由。代理软件的规则决定出口是否走 VPN；脚本只检测代理端口就绪，不识别 VPN 的内部连接状态，也不验证出口地区。

Windows 用户登录后自动运行，等待代理启动；SSH 中断后每 15 秒重试，保活失败约 45 秒后退出重建。笔记本休眠、断网或关机期间出口不可用。全天候实验需要始终在线的出口或服务器自己的网络连接。

## 手动测试（Windows PowerShell）

首次先手动 `ssh dr@10.137.11.194` 核验服务器指纹并准备密钥登录；后台任务不会接受未知主机或询问密码。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote\Start-CodexProxy.ps1
```

预览参数可加 `-CheckOnly`。端口不同时使用 `-LocalProxyPort` / `-RemoteProxyPort`，服务器包装器同步设置 `CODEX_LAPTOP_PROXY_PORT`。

## 登录后自启（Windows PowerShell）

先结束手动测试的隧道，再执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote\Set-CodexProxyAutostart.ps1
Get-ScheduledTaskInfo -TaskName Codex-Laptop-Proxy
```

脚本内容直接保存在当前用户的计划任务中，避免后台进程无法访问安装程序所见文件的问题。任务名为 `Codex-Laptop-Proxy`，使用当前用户登录权限，无需保存账户密码。启动入口使用已安装的 `pythonw.exe`（需在 PATH 中），为 PowerShell 和 SSH 指定无窗口启动及隐藏设置，因此不显示终端窗口。修改脚本或端口后重新执行安装命令。安装更新和卸载会根据进程身份清理此任务的子进程，避免留下旧 SSH 占用端口。

这是 Windows 登录后自动启动；开机但未登录时不会运行。等待本机代理启动后再连接，笔记本注销、休眠或关机会中断通道。任务失败时每分钟重启，最多 3 次；SSH 断线则由脚本持续重试。可在任务管理器中看到 pythonw、PowerShell 和 SSH 进程；隐藏窗口不能阻止手动结束这些后台进程。普通终端窗口的关闭不影响该任务。

卸载：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote\Set-CodexProxyAutostart.ps1 -Remove
```

## 服务器安装和登录

本次服务器已装 Codex，无需重装。其他机器需先安装 Node.js/npm；在代理通道建立后运行：

```bash
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=$HTTP_PROXY
npm install -g @openai/codex
```

复制包装器（Windows）：

```powershell
scp .\scripts\remote\codex-via-laptop.sh dr@10.137.11.194:/home/dr/.local/bin/codex-via-laptop
ssh dr@10.137.11.194 'chmod 700 ~/.local/bin/codex-via-laptop'
```

服务器执行：

```bash
~/.local/bin/codex-via-laptop login --device-auth
~/.local/bin/codex-via-laptop login status
cd /path/to/LLM-Password
~/.local/bin/codex-via-laptop
```

设备登录需在本机浏览器完成。若账户不支持设备码，可从本机执行 `ssh -L 1455:localhost:1455 dr@10.137.11.194`，随后在该会话执行包装器的 `login` 命令并在本机打开登录地址。本机 1455 端口必须空闲。不要把 `auth.json` 放进仓库。

只安装包装器不会自动给其他 Shell 或桌面远端会话设置代理，需显式使用包装器或配置对应进程的 HTTP(S)_PROXY。现有 OpenRouter 客户端默认忽略环境代理，采集需要代理时在私有 `.env` 中设置 `OPENROUTER_TRUST_ENV=true`。

## 本机连接远端

支持 SSH 远端项目的桌面客户端可按官方说明添加 SSH 主机；该模式仍要求远端 Codex 安装、认证及网络正确。当前本地 Codex 也可以通过执行 SSH 命令维护远端代码，这与远端运行交互式 Codex 是两种使用方式。

官方参考：[登录](https://learn.chatgpt.com/docs/auth)、[SSH 远端连接](https://learn.chatgpt.com/docs/remote-connections)。
