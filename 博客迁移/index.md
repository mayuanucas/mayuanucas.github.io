# 博客迁移


2021年1月1日，告别使用了三年的Hexo，正式迁移到Hugo.
<!--more-->

# 原因

使用Hexo+Next+GitHub Pages搭建博客已经有三年的时间了，但是随着文章数量的增长，Hexo生成博客的速度也慢下来了，而且Hexo对Latex公式的支持不给力。怀着对Golang 的信仰，至此加入Golang的生态圈，拥抱Hugo。

# 开始迁移

## 简介

> *Hugo is one of the most popular open-source static site generators. With its amazing speed and flexibility, Hugo makes building websites fun again.*

[Hugo](https://gohugo.io/)是一个基于Go语言开发的静态网站生成器，主打简单、易用、高效、易扩展、快速部署，丰富的主题也使得Hugo在个人博客站点搭建方面也使用广泛。迁移到Hugo后，安装、构建、部署整个流程相比Hexo，速度提升飞快🚀。

## 安装

本地macOS平台直接使用`Homebrew`安装

```bash
brew install hugo
```

## 创建新站点

```bash
hugo new site blog

cd blog

hugo new posts/博客迁移.md
```

这个时候就已经创建了新的博客站点`blog`，并且创建了第一篇文章`博客迁移.md`，新建的文章位于`/blog/content/posts`目录下。

{{< admonition note "新建文章注意" >}}
默认情况下，所有文章新建都为草稿，草稿文章是不渲染的，需要修改头部`draft: true`为`draft: false`
{{< /admonition >}}

## 使用主题

Hugo提供了丰富的[主题](https://themes.gohugo.io/)，可以在这选择喜欢的主题，并添加到刚刚新加的博客站点，以我选择的[LoveIt](https://github.com/dillonzq/LoveIt)主题为例
首先将主题添加到项目`blog/themes`目录，根目录下执行：

```bash
git clone -b master https://github.com/dillonzq/LoveIt.git themes/LoveIt
```

然后在`/blog/config.toml`配置主题参数：

```toml
baseURL = "http://example.org/"
# [en, zh-cn, fr, ...] 设置默认的语言
defaultContentLanguage = "zh-cn"
# 网站语言, 仅在这里 CN 大写
languageCode = "zh-CN"
# 是否包括中日韩文字
hasCJKLanguage = true
# 网站标题
title = "我的 Hugo 博客站点"

# 更改使用 Hugo 构建网站时使用的默认主题
theme = "LoveIt"

[params]
  # LoveIt 主题版本
  version = "0.2.X"

[menu]
  [[menu.main]]
    identifier = "posts"
    # 你可以在名称 (允许 HTML 格式) 之前添加其他信息, 例如图标
    pre = ""
    name = "文章"
    url = "/posts/"
    # 当你将鼠标悬停在此菜单链接上时, 将显示的标题
    title = ""
    weight = 1
  [[menu.main]]
    identifier = "tags"
    pre = ""
    name = "标签"
    url = "/tags/"
    title = ""
    weight = 2
  [[menu.main]]
    identifier = "categories"
    pre = ""
    name = "分类"
    url = "/categories/"
    title = ""
    weight = 3
```

这个主题功能很强大，更多详细配置及功能可以参考[项目Docs](https://hugoloveit.com/categories/documentation/)

## 本地展示

此时执行以下命令即可在本地 **http://localhost:1313/** 预览当前站点状态

```bash
hugo serve
```

## 快速部署

准备好部署网站时，运行

```bash
hugo
```

可以快速构建网站，项目根目录下会生成`public`目录，其中包含博客站点所有内容和资源，直接部署在web服务器即可。

以部署到github pages为例，参考[Hugo官网](https://gohugo.io/hosting-and-deployment/hosting-on-github/)说明，创建`public`子模块，关联原先github page仓库`用户名.github.io`，将每次构建结果提交到远程仓库，可以通过自动部署脚本实现快速部署

```bash
#!/bin/sh

# If a command fails then the deploy stops
set -e

printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"

# Build the project.
hugo -t LoveIt # if using a theme, replace with `hugo -t <YOURTHEME>`

# Go To Public folder
cd public

# Add changes to git.
git add .

# Commit changes.
msg="rebuilding site $(date)"
if [ -n "$*" ]; then
	msg="$*"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin master

cd ..
```

到这一步，每次更新文章之后，需要在本地 **blog** 目录下手动执行

```bash
./deploy.sh
```

来部署到github page、coding page等静态页面。

## 启用搜索

{{< version 0.2.0 >}}

基于 [Lunr.js](https://lunrjs.com/) 或 [algolia](https://www.algolia.com/), **LoveIt** 主题支持搜索功能.

### 输出配置

为了生成搜索功能所需要的 `index.json`, 请在你的 [网站配置](#site-configuration) 中添加 `JSON` 输出文件类型到 `outputs` 部分的 `home` 字段中.

```toml
[outputs]
  home = ["HTML", "RSS", "JSON"]
```

### 搜索配置

基于 Hugo 生成的 `index.json` 文件, 你可以激活搜索功能.

这是你的 [网站配置](#site-configuration) 中的搜索部分:

```toml
[params.search]
  enable = true
  # 搜索引擎的类型 ("lunr", "algolia")
  type = "algolia"
  # 文章内容最长索引长度
  contentLength = 4000
  # 搜索框的占位提示语
  placeholder = ""
  # {{< version 0.2.1 >}} 最大结果数目
  maxResultLength = 10
  # {{< version 0.2.3 >}} 结果内容片段长度
  snippetLength = 50
  # {{< version 0.2.1 >}} 搜索结果中高亮部分的 HTML 标签
  highlightTag = "em"
  # {{< version 0.2.4 >}} 是否在搜索索引中使用基于 baseURL 的绝对路径
  absoluteURL = true
  [params.search.algolia]
    index = ""
    appID = ""
    searchKey = ""
```

{{< admonition note "怎样选择搜索引擎?" >}}
以下是两种搜索引擎的对比:

* `lunr`: 简单, 无需同步 `index.json`, 没有 `contentLength` 的限制, 但占用带宽大且性能低 (特别是中文需要一个较大的分词依赖库)
* `algolia`: 高性能并且占用带宽低, 但需要同步 `index.json` 且有 `contentLength` 的限制

{{< version 0.2.3 >}} 文章内容被 `h2` 和 `h3` HTML 标签切分来提高查询效果并且基本实现全文搜索.
`contentLength` 用来限制 `h2` 和 `h3` HTML 标签开头的内容部分的最大长度.
{{< /admonition >}}

{{< admonition tip "关于 algolia 的使用技巧" >}}
你需要上传 `index.json` 到 algolia 来激活搜索功能. 你可以使用浏览器来上传 `index.json` 文件但是一个自动化的脚本可能效果更好.
[Algolia Atomic](https://github.com/chrisdmacrae/atomic-algolia) 是一个不错的选择.
为了兼容 Hugo 的多语言模式, 你需要上传不同语言的 `index.json` 文件到对应的 algolia index, 例如 `zh-cn/index.json` 或 `fr/index.json`...
{{< /admonition >}}

### 自动上传

每次写完博文都手动上传索引文件无疑是痛苦的、无意义的重复劳动。
因此我们需要把上传索引文件的操作自动化，在自动部署的时候顺便完成即可。
这里我们采用npm包 [atomic-algolia](https://www.npmjs.com/package/atomic-algolia) 来完成上传操作。

- 安装 atomic-algolia 包

  ```bash
  npm init -y // npm默认生成package.json文件
  npm install -g atomic-algolia // npm全局安装atomic-algolia
  ```

- 修改目录下的 `package.json` 文件

  ```bash
  "scripts": {
      "test": "echo \"Error: no test specified\" && exit 1",
      "algolia": "atomic-algolia"
  },
  ```

  注意 `"test"` 那一行末尾有个英文逗号，不要漏了。

- **blog** 根目录下新建 `.env` 文件

  ```bash
  ALGOLIA_APP_ID=你的Application ID
  ALGOLIA_INDEX_NAME=你的索引名字
  ALGOLIA_INDEX_FILE=public/algolia.json
  ALGOLIA_ADMIN_KEY=你的Admin API Key
  ```

  另外特别注意 `ALGOLIA_ADMIN_KEY` 可以用来管理你的索引，所以尽量不要提交到公共仓库。

- 上传索引的命令

  ```bash
  npm run algolia	// 在blog根目录下执行
  ```

  后续就是把下面的命令加到你的部署脚本即可：

  ```bash
  #!/bin/sh
  
  # If a command fails then the deploy stops
  set -e
  
  printf "\033[0;32mDeploying updates to GitHub...\033[0m\n"
  
  # Build the project.
  hugo -t LoveIt # if using a theme, replace with `hugo -t <YOURTHEME>`
  
  # Go To Public folder
  cd public
  
  # Add changes to git.
  git add .
  
  # Commit changes.
  msg="rebuilding site $(date)"
  if [ -n "$*" ]; then
  	msg="$*"
  fi
  git commit -m "$msg"
  
  # Push source and build repos.
  git push origin master
  cd ..
  # 自动更新文章索引
  npm run algolia
  ```

# 拥抱Hugo

还有更多的功能等待探索中… 目前使用下来，Hugo整体的使用体验很不错，后面会将个人文章陆续迁移到这，慢慢完善。
