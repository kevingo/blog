今天看到一篇 [Extracting data from Wikipedia using curl, grep, cut and other shell commands](http://loige.co/extracting-data-from-wikipedia-using-curl-grep-cut-and-other-bash-commands/)。作者用了一些常見的 shell commands 就從 Wikipedia 抓出他想要的資料了。這些 shell commands 包含 `curl`、`grep`、`cut`、`uniq` 和 `sort`。

有趣的是，原來 Wikipedia 只要在網址上給定 `?action=raw` 這個 query string，就可以拿到該頁面的 wikitext，這樣就會比較方便我們做資料處理。

比如說我想要抓 NBA 總得分排行榜的 wikitex，只要存取這個 url 即可：[https://en.wikipedia.org/wiki/List_of_National_Basketball_Association_career_scoring_leaders](https://en.wikipedia.org/wiki/List_of_National_Basketball_Association_career_scoring_leaders)，抓出來的結果會向下圖所示：

![image](https://github.com/kevingo/blog/raw/master/screenshot/nba-wikitext.png)

原作者想抓的資料是關於柔道的資料：[https://en.wikipedia.org/wiki/List_of_Olympic_medalists_in_judo?action=raw](https://en.wikipedia.org/wiki/List_of_Olympic_medalists_in_judo?action=raw)

組合技如下：

```
curl -sS "https://en.wikipedia.org/wiki/List_of_Olympic_medalists_in_judo?action=raw" |\
 grep -Eoi "flagIOCmedalist\|\[\[(.+)\]\]" |\
 cut -c"19-" |\
 cut -d \] -f 1 |\
 cut -d \| -f 2 |\
 sort |\
 uniq -c |\
 sort -nr
```

最後得到的結果就是奧運在柔道項目得牌的人的排名。