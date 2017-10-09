- [Golang Context Slides](http://go-talks.appspot.com/github.com/dkondratovych/golang-ua-meetup/go-context/ctx.slide#1)

- [Here are some resources you should check out if you are learning / new to Go](https://gophers.slack.com/archives/C02A8LZKT/p1492876017604943)
    - 從 slack 上看到的 Golang 學習資源

-[Go Tooling in Action](https://www.youtube.com/watch?v=uBjoTxosSys)
    - JustForFun 系列的影片，介紹一系列的 go tools
    - go list 有很多參數和隱藏用法可以看引用的路徑、名稱 ... etc
        - `go list -f '{{ .Name }}'`
    - go doc fmt: show fmt go doc in console
    - debug
    - benchmark
    - pprof
- [A Short Guide to Mastering Strings in Golang](https://mymemorysucks.wordpress.com/2017/05/03/a-short-guide-to-mastering-strings-in-golang/)
    - strings are made of up runes (not bytes)
    - characters/runes in a string are of variable length
    - ranging over a string with a for loop returns runes
    - strings are immutable while a []rune slice is mutable
- [golang official docker hub](https://hub.docker.com/_/golang/)
- [Building a Go web application deployment tool chain with Nginx, Gulp and Supervisor](https://medium.com/dev-bits/building-a-go-web-application-deployment-tool-chain-with-nginx-gulp-and-supervisor-bef6a5d454b9)
    - Gulp for minitoring the code change and update the binary
    - Supervisord for managing the process
- [go-web-workshop](https://github.com/campoy/go-web-workshop)
    - Francesc Campoy 的 golang web workshop + GCP
- [Channels](https://golang.org/doc/effective_go.html#channels)
    - Buffered V.S. Unbuffered channel
- [Pointers](https://golangbot.com/pointers/)