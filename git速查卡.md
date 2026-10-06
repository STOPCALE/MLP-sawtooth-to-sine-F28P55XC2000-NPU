# git 速查卡（Charge_Num 版）

> 目标：**离开 AI 和聊天记录，也能独立建仓、存档、推送**。建议打印一份贴桌上。
>
> 本机关键信息：账号 **STOPCALE** ｜ 提交邮箱（马甲）`317050390+STOPCALE@users.noreply.github.com` ｜ 代理 `127.0.0.1:7890`（push 前先开代理软件） ｜ 本仓库远程：`https://github.com/STOPCALE/MLP-sawtooth-to-sine-F28P55XC2000-NPU`

## 一、一句话记全流程

```
工作区（你编辑的文件） --git add--> 暂存区（打包清单） --git commit--> 本地仓库 .git（永久存档） --git push--> GitHub（云端备份）
```

## 二、新项目建仓 5 步（本地 → GitHub）

```powershell
cd 项目文件夹
git init                                  # 建仓：生成隐藏的 .git 文件夹
# ★ 先写好 .gitignore（模板见文末附录），再往下
git add .                                 # 全部文件收进打包清单
git commit -m "初始化"                     # 第一个存档点
```

GitHub 网页建仓：
1. github.com → 右上角 `+` → `New repository`
2. 填名字（可与本地文件夹不同名）→ 选 `Private`
3. 🔴 三个初始化选项（README / .gitignore / License）**一个都不勾**
4. `Create repository` → 复制页面给出的 URL

```powershell
git remote add origin <URL>               # 把云端地址记进"地址簿"
git remote -v                             # 查看已记录的地址
git push -u origin main                   # 首次上传（-u 记住上游，以后只敲 git push）
```

## 三、日常循环（每天用）

```powershell
git status            # ① 看变动（红=没收，绿=待打包）
git add .             # ② 全部收进清单（也可只收单个文件：git add 文件名）
git commit -m "说明"   # ③ 存档
git push              # ④ 上传
```

一键版（已装别名）：`git upload "说明"` = 全收 + 存档 + 上传。

## 四、status 怎么读

| 显示 | 含义 |
|---|---|
| 红字 `Untracked` / `modified` | 工作区有变动还没收 |
| 绿字 `Changes to be committed` | 已收进清单，等 commit |
| `working tree clean` | 与最新存档完全一致 |
| `up to date with 'origin/main'` | 本地与云端已同步 |

## 五、常见坑急救表

| 症状 | 原因 | 处理 |
|---|---|---|
| `fatal: not a git repository` | 没进对目录 / 没 init | `Get-Location` 看站位 |
| `Failed to connect to 127.0.0.1 port 7890` | 代理软件没开 | 打开再 push |
| `Repository not found` | 仓库没建成功 / 名字或 URL 不符 | 网页确认仓库存在，URL 一字不差 |
| `GH007 ... private email` | qq 邮箱撞隐私闸 | 换章（见第六节） |
| `! [rejected] ... fetch first` | 云端有你本地没有的提交 | 别硬推，先问 |
| 中文文件名显示成 `\351...` | git 显示怪癖 | `git config --global core.quotepath false` |
| 误删了文件 | — | `git restore 文件名`（从存档里捞回） |
| remote 地址敲错 | — | `git remote set-url origin <新URL>` |

## 六、换章急救（提交邮箱要改时）

```powershell
git commit --amend --no-edit --reset-author       # 只改"最新一条"
git rebase --root --exec "git commit --amend --no-edit --reset-author"   # 改"全部历史"
```

前提：只对**没推送过**的提交做；推过的要 force push——先问再动。

## 七、四条红线

1. **先立 `.gitignore`，再 `add .`**——大文件（权重/数据集）进了历史极难清理
2. **敲 git 前先看提示符**——命令只作用于"当前目录"的仓库
3. **删 `.git` 文件夹 = 抹掉全部历史**（文件还在，历史没了）
4. amend / rebase / reset 属于"动手术"：**没把握先问**

## 八、独立复现清单（毕业考）

全程**不看聊天记录**，只对这张卡：

- [ ] 新建文件夹，放一个 txt → `git init` → 第一次 `add` / `commit`
- [ ] GitHub 建一个**空仓**（不勾初始化）→ `remote add` → `push -u`
- [ ] 改一次文件，走一遍日常循环 → 网页能看到第 2 条提交
- [ ] `git status` 显示 `up to date`

全过 = 出师。

## 附录：.gitignore 模板（Python 项目）

```
__pycache__/
*.py[cod]
.venv/
venv/
.vscode/
.idea/
Thumbs.db
desktop.ini
# 想忽略生成物时，取消下面注释：
# *.png
# *.pt
# *.onnx
```
