# Using the Docker environment

- With Docker, there is no need to pollute the environment with Anaconda.

## Our Docker Environment

- We use Docker and docker-compose
  - [Docker Installation Document](https://docs.docker.com/install/)
  - [Docker Compose Installation Document](https://docs.docker.com/compose/install/)

```bash
$ docker version
Client:
 Version:           18.06.1-ce
 API version:       1.38
 Go version:        go1.10.3
 Git commit:        e68fc7a
 Built:             Tue Aug 21 17:21:31 2018
 OS/Arch:           darwin/amd64
 Experimental:      false

Server:
 Engine:
  Version:          18.06.1-ce
  API version:      1.38 (minimum version 1.12)
  Go version:       go1.10.3
  Git commit:       e68fc7a
  Built:            Tue Aug 21 17:29:02 2018
  OS/Arch:          linux/amd64
  Experimental:     true

$ docker-compose -v
docker-compose version 1.22.0, build f46880f
```

## How to use the Docker environment

### In case of using docker-compose

```bash
$ git pull git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy
$ docker-compose -f docker-compose.yml up -d --build
$ docker-compose exec kcfconvoy bash
```

### In case of using only Docker

#### In case of using `docker build`

```bash
$ git pull git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy
$ docker build -t kcfconvoy:latest .
$ docker run -it -p 8888:8888 --name kcfconvoy kcfconvoy:latest bash
```

#### In case of using `docker pull`

```bash
$ docker pull suecharo/kcfconvoy:latest .
$ docker run -it -p 8888:8888 --name kcfconvoy suecharo/kcfconvoy:latest bash
```

## Develop environment

- We have prepared an environment for debug and development
- The difference from the environment mentioned above is as follows:
  - The directory of `kcfconvoy` is mounted on `/opt/kcfconvoy` on the docker side
  - Since `/opt/kcfconvoy` is written in `PYTHONPATH`, the changed part of source is immediately reflected

```bash
$ git pull git@github.com:KCF-Convoy/kcfconvoy.git
$ cd kcfconvoy
$ docker-compose -f docker-compose.dev.yml up -d --build
$ docker-compose -f docker-compose.dev.yml exec kcfconvoy bash
```

## Jupyter notebook

- How to start jupyter in docker container
  - Execute the following command
  - In local browser, use Token to access `localhost:8888`

```bash
$ docker-compose -f docker-compose.dev.yml up -d --build
$ docker-compose -f docker-compose.dev.yml exec kcfconvoy bash
$ jupyter notebook --ip='0.0.0.0' --allow-root
```
