cp -r ../../tsclient/olestole/Downloads/$1 ./data/images
ns-train nerfacto --data data/images/$1 --viewer.websocket-port 7009
