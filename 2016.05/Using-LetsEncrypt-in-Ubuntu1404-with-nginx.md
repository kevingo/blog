相信大家都知道 [Let's Encrypt](https://letsencrypt.org/) 是一個免費提供SSL 憑證的 CA (certificate authority) ，過去要申請 SSL 的憑證，往往需要每年花一筆錢向其他的認證公司來申請，而且光是填寫申請表，到憑證核發下來，可能都過了好幾天了，現在有 [Let's Encrypt](https://letsencrypt.org/)  這樣透過 Shell 就可以直接核發憑證的機構出現，大家還有什麼理由不將自己管理的網站加上 SSL 呢？

這裏記錄一下透過  [Let's Encrypt](https://letsencrypt.org/)  官方的工具來申請憑證的過程給大家參考。這裏我是用 Nginx 當成掛載 SSL 憑證的 web server。

###  1. 首先要下載官方的工具

```
$ git clone https://github.com/letsencrypt/letsencrypt
$ cd letsencrypt
$ ./letsencrypt-auto --help
```

如果成功的話，運行 `./letsencrypt-auto --help` 這個指令應該會看到類似以下的內容：

```
Checking for new version...
Requesting root privileges to run letsencrypt...
   sudo /home/ubuntu/.local/share/letsencrypt/bin/letsencrypt --help

  letsencrypt-auto [SUBCOMMAND] [options] [-d domain] [-d domain] ...

The Let's Encrypt agent can obtain and install HTTPS/TLS/SSL certificates.  By
default, it will attempt to use a webserver both for obtaining and installing
the cert. Major SUBCOMMANDS are:

  (default) run        Obtain & install a cert in your current webserver
  certonly             Obtain cert, but do not install it (aka "auth")
  install              Install a previously obtained cert in a server
  renew                Renew previously obtained certs that are near expiry
  revoke               Revoke a previously obtained certificate
  rollback             Rollback server configuration changes made during install
  config_changes       Show changes made to server config during installation
  plugins              Display information about installed plugins

Choice of server plugins for obtaining and installing cert:

  --apache          Use the Apache plugin for authentication & installation
  --standalone      Run a standalone webserver for authentication
  (nginx support is experimental, buggy, and not installed by default)
  --webroot         Place files in a server's webroot folder for authentication
```

### 2. 產生憑證

[Let's Encrypt](https://letsencrypt.org/) 內建了 server plugins 讓你可以快速地根據你的 server 種類來產生憑證，可以看到有 apache、standalone 或 webroot 的方式：的方式：

```
--apache          Use the Apache plugin for authentication & installation
--standalone      Run a standalone webserver for authentication
  (nginx support is experimental, buggy, and not installed by default)
--webroot         Place files in a server's webroot folder for authentication
```
要注意的是，如果你用 apache 或 standalone 的方式，會需要使用 80 port，如果你的 web server 可以停下來當然沒問題，不然就可以改用 webroot 的方式來驗證。

在這裡我指定用 webroot 的方式進行驗證，驗證的目錄就定在 /etc/nginx/html 下，至於 `-d`  的參數則是用來指定 SSL 憑證所作用的 domain name。`certonly` 則是說，我只要產生憑證而已，不要亂搞我的 web server 任何的參數。

```
./letsencrypt-auto certonly -a webroot --webroot-path=/etc/nginx/html -d example.com -d blog.example.com
```

成功之後，應該會看到類似的訊息：
```
IMPORTANT NOTES:
 - Congratulations! Your certificate and chain have been saved at
   /etc/letsencrypt/live/example.com/fullchain.pem. Your cert will
   expire on 2016-08-05. To obtain a new version of the certificate in
   the future, simply run Let's Encrypt again.
```

### 3. 將 certificate 和 key 放到 Nginx 設定中

```
server {
  ...

  ssl_certificate /etc/letsencrypt/live/xxxx/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/xxxx/privkey.pem;

  ...
}


```
預設的憑證會放在 `/etc/letsencrypt/live/$domain` 下。


### 4. 定期更新 Certificate

[Let's Encrypt](https://letsencrypt.org/) 的憑證每隔三個月會過期，記得要去更新才行。更新的方式只要每隔一段時間執行下述指令即可：

```
/opt/letsencrypt/letsencrypt-auto renew
```

當然你也可以寫到 cron job 中。
