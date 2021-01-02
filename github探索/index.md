# Github入门与实践


本文对 GitHub 的使用进行介绍.

<!-- more -->

# 前言

GitHub是通过Git进行版本控制的软件源代码托管服务平台，于2008年4月上线，2018年6月被微软公司收购。GitHub同时提供付费账户和免费账户，这两种账户都可以创建公开或私有的代码仓库，但付费用户支持更多功能。除了允许个人或组织创建和访问保管中的代码以外，GitHub还提供了一些方便共同开发软件的功能，例如：允许用户追踪其他用户、组织、软件库的动态，对软件代码的改动或bug提出评论等。GitHub还提供了图表功能，用于显示开发者在代码库上工作以及软件项目的开发活跃程度。

GitHub 2019年度报告显示，在过去的一年中，GitHub新增了一千万用户，现在总共有超过四千万用户，GitHub上的仓库数量超过 1 亿。

# Git与GitHub的区别和联系

谈到 GitHub, 就必定会提到Git。 Git是一个开源的分布式版本控制系统,简单来说，Git 是一个管理"代码的历史记录"的工具,而 GitHub 本质上是一个代码托管平台，它提供的是基于 Git 的代码托管服务。对于一个团队来说，即使不使用 GitHub，也可以通过自己搭建和管理 Git 服务器来进行代码库的管理，或者选择其它一些代码托管平台，如 Gitee(码云), GitLab等。

使用 Git 管理代码前需要从 Git 官网 **https://git-scm.com/** 下载相应平台的安装包并完成安装. Git 的本身不具备图形界面，一般只能在终端输入命令进行使用.但是在安装 Git 的同时，其实也装好了它提供的可视化工具，gitk 和 git-gui. gitk 是一个历史记录的图形化查看器, git-gui 主要是一个用来制作提交的工具.

{% asset_img github探索/git.png Git官网 %}

GitHub 也发布了面向工作流程的 Git 客户端：提供了Windows 版和 Mac 版,它们很好的展示了一个面向工作流程的工具——专注于提升常用的功能及提高协作效率.

{% asset_img github_desktop.png GitHub客户端 %}

# GitHub概览

## 基础应用场景

GitHub 的基础应用场景是作为远程的代码存储,代码版本控制.

## 常用应用场景

### 协同开发

{% asset_img xietongkaifa.jpeg 多人协同开发 %}

### 获取(学习)优秀的开源项目

由于存放在 Github **公有仓库**的代码是公开的，所以可以很方便的获取、使用、学习这些优秀开源项目的代码和文档.

#### 国内外科技公司

| 国外      | GitHub地址                   | 国内     | GitHub地址                 |
| --------- | ---------------------------- | -------- | -------------------------- |
| Google    | https://github.com/google    | 阿里巴巴 | https://github.com/alibaba |
| Facebook  | https://github.com/facebook  | 腾讯     | https://github.com/Tencent |
| Microsoft | https://github.com/microsoft | 滴滴     | https://github.com/didi    |
|           |                              |          |                            |

#### 世界闻名的技术专家

1. Linux 发明者 `Linus Torvalds`：https://github.com/torvalds

{% asset_img LinusTorvalds.png Linus Torvalds %}

2. Hands-On Machine Learning with Scikit-Learn and TensorFlow 的作者 `Aurélien Geron`：https://github.com/ageron

{% asset_img hands_on_author.png Aurélien Geron %}

#### 优秀的开源项目

| 项目         | GitHub地址                                   |
| ------------ | -------------------------------------------- |
| TensorFlow   | https://github.com/tensorflow/tensorflow     |
| scikit-learn | https://github.com/scikit-learn/scikit-learn |
|      pytorch | https://github.com/pytorch/pytorch           |
|  |                                              |

## 其它应用场景

### 搭建个人网站

基于 `GitHub Pages` 搭建博客，不仅搭建简单，同时还可自定义样式、绑定域名。

{% asset_img luc.png 个人博客 %}

### 接触优秀开发者的有效渠道

GitHub 个人主页会有联系邮箱、个人网站等信息，通过这些信息可以与技术专家进行沟通交流。

{% asset_img hands_on_author.png Aurélien Geron %}

### 用GitHub协作翻译

国内的一些社区会用 GitHub 组织志愿者进行文档的协作翻译。

{% asset_img sklearn_cn.png scikit-learn 官方文档中文版 %}

# 个人仪表板

个人仪表板是登录 GitHub 时显示的第一页。访问个人仪表板，可以跟踪参与或关注的议题和拉取请求，浏览常用仓库和团队页面，了解订阅的组织和仓库中近期活动的最新信息，以及探索推荐的仓库。登录后要访问个人仪表板，在任意页面点击左上角的网站图标就能跳转到仪表板页面。

{% asset_img home.png 个人仪表板%}

# 个人资料(Profile)

个人资料页面不仅展示了开发者的个人介绍, 联系邮箱, 博客地址, 社交账号, 还展示了开发者创建的或 Fork 的仓库, 页面最下方还展示了开发者每天的活跃程度(每天提交的 commit 越多, 对应日期的小方格颜色越深).

{% asset_img profile.png  your profile %}

# 仓库

**GitHub 常用术语介绍:**

- **Repository**：简称Repo，可以理解为“仓库”，我们的项目就存放在仓库之中。也就是说，如果我们想要建立项目，就得先建立仓库；有多个项目，就建立多个仓库。

- **Issues**：可以理解为“问题”，举一个简单的例子，如果我们开源一个项目，如果别人看了我们的项目，并且发现了bug，或者感觉那个地方有待改进，他就可以给我们提出Issue，等我们把Issues解决之后，就可以把这些Issues关闭；反之，我们也可以给他人提出Issue。

- **Watch**：可以理解为“观察”，如果我们Watch了一个项目，之后，如果这个项目有了任何更新，我们都会在第一时候收到该项目的更新通知。

- **Star**：可以理解为“点赞”，当我们感觉某一个项目做的比较好之后，就可以为这个项目点赞，而且我们点赞过的项目，都会保存到我们的Star之中，方便我们随时查看。在 GitHub 之中，如果一个项目的点星数能够超百，那么说明这个项目已经很不错了。

- **Fork**：可以理解为“拉分支”，如果我们对某一个项目比较感兴趣，并且想在此基础之上开发新的功能，这时我们就可以Fork这个项目，这表示复制一个完成相同的项目到我们的 GitHub 账号之中，而且独立于原项目。之后，我们就可以在自己复制的项目中进行开发了。

- **Pull Request**：可以理解为“提交请求”，此功能是建立在Fork之上的，如果我们Fork了一个项目，对其进行了修改，而且感觉修改的还不错，我们就可以对原项目的拥有者提出一个Pull请求，等其对我们的请求审核，并且通过审核之后，就可以把我们修改过的内容合并到原项目之中，这时我们就成了该项目的贡献者。

- **Merge**：可以理解为“合并”，如果别人Fork了我们的项目，对其进行了修改，并且提出了Pull请求，这时我们就可以对这个Pull请求进行审核。如果这个Pull请求的内容满足我们的要求，并且跟我们原有的项目没有冲突的话，就可以将其合并到我们的项目之中。当然，是否进行合并，由我们决定。

- **Gist**：如果我们没有项目可以开源或者只是单纯的想分享一些代码片段的话，我们就可以选择Gist。

{% asset_img repo.png  仓库 %}

# Topics

GitHub Topic 页面展示了最新和最流行的讨论主题，在这里不仅能够看到开发项目，还能看到很多非开发技术的讨论主题.

{% asset_img topic.png  Topics %}

# Trending

GitHub Trending 页面展示了每天/每周/每月周期的热门 Repositories 和 Developers，可以看到在某个周期处于热门状态的开发项目和开发者。

{% asset_img trending.png  Trending %}

# GitHub搜索

{% asset_img search01.png  GitHub搜索 %}

{% asset_img search02.png  GitHub搜索 %}

## 搜索开发者

{% asset_img skaifazhe.jpg 搜索开发者 %}

比如需要寻找国产软件，搜索时设置 location 为 china，如果要寻找使用 javascript 语言开发者，则再增加 language 为 javascript，整个搜索条件就是：`language:javascript location:china`，从搜索结果来看，我们找到了超过 2.1 万名地区信息填写为 china 的 javascript 开发者，朋友们熟悉的阮一峰老师排在前列。根据官方手册，搜索 GitHub 用户时还支持使用 followers、in:fullname 组合条件进行搜索。

{% asset_img kfz.png 搜索开发者 %}

## 搜索仓库

在 GitHub 上找到优秀的项目和工具，通过关键字或者设置搜索条件能够帮助我们事半功倍找到好资源。

{% asset_img sxm01.jpg 搜索仓库 %}

### 设定搜索条件

如果明确需要寻找某类特定的项目，比如用某种语言开发、Stars 数量需要达到标准的项目，在搜索框中直接输入搜索条件即可。其中用于发现项目，常用的搜索条件有：stars:、language:、forks:、in:，这些条件是设置搜索条件为项目收藏数量、开发语言、Fork数量. 比如输入 stars:>=5000 language:python，得到的结果 就是收藏大于和等于 5000 的 python 项目。

{% asset_img tj01.png 搜索条件 %}

通过 in: 限定符，可以将搜索限制为仓库名称、仓库说明、自述文件内容或这些的任意组合。



| 限定符         | 示例                                                         |
| -------------- | ------------------------------------------------------------ |
| in:name        | **python in:name** 匹配其名称中含有 "python" 的仓库。        |
| in:description | **python in:name,description** 匹配其名称或说明中含有 "python" 的仓库。 |
| in:readme      | **python in:readme** 匹配其自述文件中提及 "python" 的仓库。  |
|                |                                                              |

{% asset_img tj02.png in限定符 %}

如果觉得记住这些搜索条件略显繁琐的话，使用 GitHub 提供的高级搜索功能同样可自定义条件进行搜索。或者参考官方给出的帮助指南 [**Searching on GitHub**](https://help.github.com/en/github/searching-for-information-on-github/searching-on-github) ，里面有更多关于项目、代码、评论、问题等搜索技巧。

{% asset_img help.png 指南 %}

# 结语

GitHub 网站上有很多优秀的开源项目，利用 GitHub 提供的各种功能，包括高级搜索、Topic、Trending 等专题页面，不仅可以帮助我们发现更多好用的效率工具和开源项目, 而且还能帮助我们了解业界最新的研究动态, 提高开发能力。
