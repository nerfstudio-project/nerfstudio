# Environment

# Installing docker

#### Troubleshooting

```
# https://forums.docker.com/t/failing-to-start-dockerd-failed-to-create-nat-chain-docker/78269
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
sudo service docker start

VERSION_STRING=18.03.1~ce~3-0~ubuntu
sudo apt-get install docker-ce=$VERSION_STRING docker-ce-cli=$VERSION_STRING containerd.io

# https://github.com/docker/for-linux/issues/123#issuecomment-346546953
sudo ip link add name docker0 type bridge
sudo ip addr add dev docker0 172.17.0.1/16
sudo dockerd --debug --iptables=false
```