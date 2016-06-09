透過 Dockerfile 來產生 docker image 相信大家都很熟悉，不過要怎麼撰寫一個夠好的 Dockerfile 呢？在 Docker 的官方網站上，有一份 [Dockerfile Best Practice(https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/)，看過之後，你就會知道要怎麼寫了。

在這裡摘錄幾個重點給大家參考：

#### 1. 撰寫 .dockerignore 檔案
把不需要包在 image 當中的檔案給排除，跟 .gitignore 的意思是類似的。

#### 2. 避免安裝不需要的套件

#### 3. 在一個 container 中，只執行單一個 process
這是為了要讓你的每個 docker container 可以解耦，也避免讓一個 container 過於龐大和複雜。如果需要有多個服務之間有相依性的話，可以使用 container link 的方式。

#### 4. 最小化你的 layes
在「可讀性」和「image size」之間取得平衡。

#### 5. 如果有多行參數時，記得按照字母排序
這裡的建議是，如果把多個參數放在一行時，可以將其透過字母排序。這樣的好處是方便閱讀，別人如果要 PR 你的 Dockerfile 時也比較容易。

例如在安裝套件時，將套件名稱排序好：

```
RUN apt-get update && apt-get install -y \
  bzr \
  cvs \
  git \
  mercurial \
  subversion
```

#### 6. 使用 cache
Docker daemon 在建構 image 時，是透過 Dockerfile 中一行一行的指令來建構一層層的 layer，每執行一行就會建構一層中繼的 image。如果你指令是按照固定順序，那 Docker daemon 就會去找找看 cache 中有沒有可以用的 image，而不用重新建構，這樣就會加速 docker build 的速度，如果不想要它從 cache 拿，你也可以在 build 的時候給定 `--no-cache=true` 的參數。

#### 7. 撰寫 Dockerfile 的建議

- 使用官方的 image
- 讓 `RUN` 指令的可讀性、可理解性更好。比如說你可以把複雜的指令拆解成多行，然後用 `\` 隔開
- 謹慎地使用 `apt-get` 指令。在這裡官方建議不要使用 `RUN apt-get upgrade` 或 `dist-upgrade`，如果要更新某個軟體的版本時，請留給 Base Image 那一層去做。原因是，當你要升級某個軟體的版本時，可能會因為沒有權限去修改 base image 的權限，而導致升級失敗。
- 將 `RUN apt-get undate` 和 `RUN apt-get install` 寫在同一行，例如：


```
RUN apt-get update && apt-get install -y \
curl
```

如果你分開寫：

```
    FROM ubuntu:14.04
    RUN apt-get update
    RUN apt-get install -y curl
```

我們知道每一個指令在 Dockerfile 中都會被 cache 起來，所以當你想要再安裝別的套件，所以你修改 Dockerfile 變成：

```
    FROM ubuntu:14.04
    RUN apt-get update
    RUN apt-get install -y nginx
```

Docker 會發現 `RUN apt-get update` 這一行之前和前一個 Dockerfile 中一模一樣，所以就不會執行了。