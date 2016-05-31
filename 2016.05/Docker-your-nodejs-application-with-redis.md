這篇主要紀錄如何將你的 Nodejs 應用程式和 Redis 都包成 Docker，並且彼此之間可以互相溝通。

### 將 Redis 透過 Docker 打包

1.參考 [docker-redis](https://github.com/kevingo/docker-redis) 的步驟，建立一個 Redis 的 docker image，並且啟動它。如果成功的話，執行 `docker ps` 應該會看到 redis 的 container 被正確啟動：

![image](https://github.com/kevingo/blog/raw/master/screenshot/docker-redis.png)

### 撰寫 Node.js 程式並和 Redis 溝通

2.撰寫你自己的 Node.js 程式，在 `redis.createClient('6379', 'redis');` 中，我們用 redis 取代一個固定的位置，等等我們會用 `link` redis container 的方式來連接我們在步驟一所啟動的 redis container：

```
var redis = require('redis');
var client = redis.createClient('6379', 'redis');

client.set('foo', 'bar');

client.get('foo', function(err, reply) {
	console.log(reply);
});

client.quit();
```

3.撰寫 Dockerfile：

```
FROM node:argon

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY package.json /usr/src/app/
RUN npm install

COPY . /usr/src/app

CMD ["npm", "start"]
```

4.將上述的程式打包成 image：

```
docker build -t node-redis-docekr .

```

5.透過 `--link` 的方式啟動上述步驟建立的 container，並且連接到第一步驟所建立的 redis server container：

```
docker run -d --name node-redis-docker --link redis:redis node-redis-docker

```

6.你可以用 `docker logs` 的指令來觀看產生的結果：

```
docker logs node-redis-docker
```

![image](https://github.com/kevingo/blog/raw/master/screenshot/docker-logs.png)

7.如果懶得自己寫，也可以抓打包好的程式： [node-redis-docker](https://github.com/kevingo/node-redis-docker)
